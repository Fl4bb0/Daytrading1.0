from __future__ import annotations

import unittest
from typing import cast

import requests

from kvant.kdata.alpha_vantage_retriever import AlphaVantagePlanError, get_intraday_month


class _FakeResponse:
    def __init__(self, text: str, json_payload: object | None = None):
        self.text = text
        self._json_payload = json_payload

    def raise_for_status(self) -> None:
        return None

    def json(self):
        return {} if self._json_payload is None else self._json_payload


class _FakeSession:
    def __init__(self, text: str, json_payload: object | None = None):
        self._text = text
        self._json_payload = json_payload
        self.last_kwargs = None

    def get(self, *_args, **_kwargs):
        self.last_kwargs = _kwargs
        return _FakeResponse(self._text, self._json_payload)


class AlphaVantageRetrieverTests(unittest.TestCase):
    def test_request_uses_documented_intraday_parameters(self) -> None:
        payload = {
            "Meta Data": {},
            "Time Series (1min)": {
                "2026-03-12 09:30:00": {
                    "1. open": "100.0",
                    "2. high": "101.0",
                    "3. low": "99.5",
                    "4. close": "100.5",
                    "5. volume": "12345",
                },
            },
        }
        session = _FakeSession("{}", payload)
        get_intraday_month(
            "AAPL",
            "2026-03",
            apikey="demo",
            session=cast(requests.Session, cast(object, session)),
        )
        assert session.last_kwargs is not None
        params = session.last_kwargs["params"]
        self.assertEqual(params["function"], "TIME_SERIES_INTRADAY")
        self.assertEqual(params["symbol"], "AAPL")
        self.assertEqual(params["interval"], "1min")
        self.assertEqual(params["month"], "2026-03")
        self.assertEqual(params["outputsize"], "full")
        self.assertEqual(params["datatype"], "json")
        self.assertEqual(params["adjusted"], "true")
        self.assertEqual(params["extended_hours"], "true")
        self.assertEqual(params["apikey"], "demo")

    def test_parse_intraday_month_json(self) -> None:
        payload = {
            "Meta Data": {},
            "Time Series (1min)": {
                "2026-03-12 09:31:00": {
                    "1. open": "100.5",
                    "2. high": "101.2",
                    "3. low": "100.1",
                    "4. close": "101.0",
                    "5. volume": "4567",
                },
                "2026-03-12 09:30:00": {
                    "1. open": "100.0",
                    "2. high": "101.0",
                    "3. low": "99.5",
                    "4. close": "100.5",
                    "5. volume": "12345",
                },
            },
        }
        df = get_intraday_month(
            "AAPL",
            "2026-03",
            apikey="demo",
            session=cast(requests.Session, cast(object, _FakeSession("{}", payload))),
        )
        self.assertFalse(df.empty)
        self.assertEqual(list(df.columns), ["open", "high", "low", "close", "volume"])
        self.assertEqual(str(df.index.tz), "UTC")

    def test_parse_intraday_month_csv(self) -> None:
        csv_body = (
            "time,open,high,low,close,volume\n"
            "2026-03-12 09:30:00,100.0,101.0,99.5,100.5,12345\n"
            "2026-03-12 09:31:00,100.5,101.2,100.1,101.0,4567\n"
        )
        session = _FakeSession(csv_body)
        df = get_intraday_month(
            "AAPL",
            "2026-03",
            apikey="demo",
            datatype="csv",
            session=cast(requests.Session, cast(object, session)),
        )
        self.assertFalse(df.empty)
        self.assertEqual(list(df.columns), ["open", "high", "low", "close", "volume"])
        self.assertEqual(str(df.index.tz), "UTC")
        assert session.last_kwargs is not None
        self.assertEqual(session.last_kwargs["params"]["datatype"], "csv")

    def test_plan_error_raises_specific_exception(self) -> None:
        payload = {
            "Information": "Thank you for using Alpha Vantage! This is a premium endpoint. You may subscribe.",
        }
        with self.assertRaises(AlphaVantagePlanError):
            get_intraday_month(
                "AAPL",
                "2026-03",
                apikey="demo",
                session=cast(requests.Session, cast(object, _FakeSession("{}", payload))),
            )


if __name__ == "__main__":
    unittest.main()

