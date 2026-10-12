import base64
import contextlib
import datetime as dt
import io
import json
import unittest
from unittest.mock import Mock, patch

from scripts import refresh_dhan_token as refresh
from scripts import run_agent1 as agent
from src.delivery import dispatch, telegram
import pandas as pd
from nacl import public, encoding

class RecoveryTests(unittest.TestCase):
    def setUp(self):
        # Any accidental network call is an immediate test failure.
        self.network = patch('requests.sessions.Session.request', side_effect=AssertionError('Network forbidden'))
        self.network.start()
        self.addCleanup(self.network.stop)

    def env(self, missing=None, expired=False):
        exp = int((dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=-1 if expired else 12)).timestamp())
        token = 'fake.' + base64.urlsafe_b64encode(json.dumps({'exp': exp}).encode()).decode() + '.fake'
        values = {'DHAN_ACCESS_TOKEN': token, 'DHAN_CLIENT_ID': 'fake-client', 'GH_TOKEN': 'fake-gh', 'GITHUB_REPOSITORY': 'owner/repo'}
        if missing: values.pop(missing)
        def get(key, default=None, required=False):
            if required and key not in values: raise RuntimeError('Missing ' + key)
            return values.get(key, default)
        return patch.object(refresh, 'get_env', side_effect=get)

    def test_missing_persistence_settings_never_renew(self):
        for key in ('GH_TOKEN', 'GITHUB_REPOSITORY'):
            with self.subTest(key=key), self.env(missing=key), patch.object(refresh, 'renew') as renew:
                with self.assertRaises(RuntimeError): refresh.main()
                renew.assert_not_called()

    def test_expired_never_renew(self):
        with self.env(expired=True), patch.object(refresh, 'renew') as renew:
            with self.assertRaisesRegex(RuntimeError, 'EXPIRED'): refresh.main()
            renew.assert_not_called()

    def test_preflight_failure_never_renew(self):
        with self.env(), patch.object(refresh, 'github_public_key', side_effect=RuntimeError('HTTP 403')), patch.object(refresh, 'renew') as renew:
            with self.assertRaisesRegex(RuntimeError, '403'): refresh.main()
            renew.assert_not_called()

    def test_check_has_no_network_or_renewal(self):
        with self.env(): refresh.main(check=True)

    def test_renewal_order_and_no_token_output(self):
        calls = []
        with self.env(), patch.object(refresh, 'github_public_key', side_effect=lambda *a: calls.append('preflight') or {'key': 'fake'}), patch.object(refresh, 'renew', side_effect=lambda *a: calls.append('renew') or 'PRIVATE-REPLACEMENT'), patch.object(refresh, 'update_github_secret', side_effect=lambda *a, **kw: calls.append('persist')), contextlib.redirect_stdout(io.StringIO()) as output:
            refresh.main()
            self.assertEqual(calls, ['preflight', 'renew', 'persist'])
            self.assertNotIn('PRIVATE-REPLACEMENT', output.getvalue())

    def test_secret_encryption_and_failure_redaction(self):
        private = public.PrivateKey.generate()
        key = {'key': private.public_key.encode(encoding.Base64Encoder()).decode(), 'key_id': 'test-id'}
        with patch.object(refresh.requests, 'put', return_value=Mock(status_code=204)) as put:
            refresh.update_github_secret('NAME', 'PRIVATE-REPLACEMENT', 'owner/repo', 'fake', key)
            encrypted = put.call_args.kwargs['json']['encrypted_value']
            self.assertEqual(public.SealedBox(private).decrypt(base64.b64decode(encrypted)), b'PRIVATE-REPLACEMENT')
        with patch.object(refresh.requests, 'put', return_value=Mock(status_code=403, text='PRIVATE-REPLACEMENT')):
            with self.assertRaises(RuntimeError) as error: refresh.update_github_secret('NAME', 'PRIVATE-REPLACEMENT', 'owner/repo', 'fake', key)
            self.assertNotIn('PRIVATE-REPLACEMENT', str(error.exception))

    def key(self):
        private = public.PrivateKey.generate()
        return {'key': private.public_key.encode(encoding.Base64Encoder()).decode(), 'key_id': 'test-id'}

    def test_transient_write_retries_same_payload_and_renews_once(self):
        import requests
        responses = [requests.Timeout('PRIVATE-TOKEN'), Mock(status_code=429), Mock(status_code=503), Mock(status_code=204)]
        with self.env(), patch.object(refresh, 'github_public_key', return_value=self.key()), patch.object(refresh, 'renew', return_value='PRIVATE-REPLACEMENT') as renew, patch.object(refresh.requests, 'put', side_effect=responses) as put, patch.object(refresh.time, 'sleep') as sleep, contextlib.redirect_stdout(io.StringIO()) as output:
            refresh.main()
            renew.assert_called_once()
            self.assertEqual(put.call_count, 4)
            payloads = [call.kwargs['json'] for call in put.call_args_list]
            self.assertTrue(all(payload == payloads[0] for payload in payloads))
            self.assertEqual([call.args[0] for call in sleep.call_args_list], [2, 4, 8])
            self.assertNotIn('PRIVATE', output.getvalue())

    def test_exhausted_write_retries_are_bounded_and_redacted(self):
        import requests
        for response in (requests.Timeout('PRIVATE-TOKEN'), Mock(status_code=500, text='PRIVATE-TOKEN')):
            with self.subTest(response=type(response).__name__), patch.object(refresh.requests, 'put', side_effect=[response] * 4) as put, patch.object(refresh.time, 'sleep') as sleep:
                with self.assertRaisesRegex(RuntimeError, 'not confirmed') as error:
                    refresh.update_github_secret('NAME', 'PRIVATE-REPLACEMENT', 'owner/repo', 'fake', self.key())
                self.assertEqual(put.call_count, 4)
                self.assertEqual(sleep.call_count, 3)
                self.assertNotIn('PRIVATE', str(error.exception))
                self.assertTrue(error.exception.__suppress_context__)

    def test_nonretryable_write_stops_without_sleep(self):
        for status in (400, 401, 403, 404, 422):
            with self.subTest(status=status), patch.object(refresh.requests, 'put', return_value=Mock(status_code=status, text='PRIVATE')) as put, patch.object(refresh.time, 'sleep') as sleep:
                with self.assertRaisesRegex(RuntimeError, str(status)) as error:
                    refresh.update_github_secret('NAME', 'PRIVATE-REPLACEMENT', 'owner/repo', 'fake', self.key())
                put.assert_called_once()
                sleep.assert_not_called()
                self.assertNotIn('PRIVATE', str(error.exception))

    def test_dispatch_false_and_fallback(self):
        bad, good = Mock(), Mock()
        bad.is_configured.return_value = good.is_configured.return_value = True
        bad.send.return_value = False
        good.send.return_value = True
        with patch.object(dispatch, '_CHANNELS', [('bad', bad)]): self.assertFalse(dispatch.send('body'))
        with patch.object(dispatch, '_CHANNELS', [('bad', bad), ('good', good)]): self.assertTrue(dispatch.send('body'))
        good.send.assert_called_once()

    def test_failed_delivery_has_no_receipt_then_retry_records_one(self):
        with patch.object(agent.dispatch, 'send', side_effect=[False, True]), patch.object(agent, 'log_delivery') as receipt:
            with self.assertRaises(RuntimeError): agent._deliver_report('report', pd.Timestamp('2026-10-12'))
            receipt.assert_not_called()
            agent._deliver_report('report', pd.Timestamp('2026-10-12'))
            self.assertEqual(receipt.call_args.args[0]['date'], '2026-10-12')

    def test_dry_run_error_never_alerts(self):
        with patch.object(agent.dispatch, 'send') as send:
            agent._alert_failure(RuntimeError('PRIVATE'), dry_run=True)
            send.assert_not_called()
            agent._alert_failure(RuntimeError('PRIVATE'), dry_run=False)
            self.assertNotIn('PRIVATE', str(send.call_args))

    def test_backup_guard_uses_receipts_not_predictions(self):
        date = pd.Timestamp('2026-10-12')
        with patch.object(agent, 'DhanClient'), patch.object(agent, 'update_opening_range_cache'), patch.object(agent, '_latest_session', return_value=(date, {'open': 25000})), patch.object(agent, 'read_jsonl', return_value=pd.DataFrame([{'date': '2026-10-12'}])) as read, patch.object(agent, '_daily_frame') as daily:
            self.assertEqual(agent.run(), '')
            read.assert_called_once_with(agent.DELIVERIES)
            daily.assert_not_called()
        with patch.object(agent, 'DhanClient'), patch.object(agent, 'update_opening_range_cache'), patch.object(agent, '_latest_session', return_value=(date, {'open': 25000})), patch.object(agent, 'read_jsonl', return_value=pd.DataFrame()) as read, patch.object(agent, '_daily_frame', side_effect=RuntimeError('reached prediction pipeline')):
            with self.assertRaisesRegex(RuntimeError, 'reached prediction pipeline'): agent.run()
            read.assert_called_once_with(agent.DELIVERIES)

    def test_telegram_exception_does_not_print_token_url(self):
        import requests
        with patch.object(telegram.requests, 'post', side_effect=requests.ConnectionError('https://api.telegram.org/botPRIVATE/sendMessage')), patch.object(telegram.time, 'sleep'), contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertFalse(telegram._send_chunk('fake', 'fake', 'body'))
            self.assertNotIn('PRIVATE', output.getvalue())

if __name__ == '__main__': unittest.main()
