"""Daily-high PDF: path-max preferred over the (warm-biased) product of horizon CDFs."""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from kalshi_api import algorithm as alg  # noqa: E402


def _model(high_temps, high_probs):
    return {"temps": high_temps, "probs": high_probs, "high_temps": high_temps, "high_probs": high_probs}


def test_path_high_pdf_mixes_3h_pair_with_weights_and_normalises():
    results = {
        "accuweather_3h": _model([88, 89], [0.5, 0.5]),
        "var_3h": _model([89, 90], [1.0, 0.0]),
    }
    pdf = alg._get_path_high_pdf(results)
    assert sum(pdf.values()) == pytest.approx(1.0)
    wa, wv = alg.HORIZON_WEIGHTS["3h"]["accuweather_3h"], alg.HORIZON_WEIGHTS["3h"]["var_3h"]
    assert pdf[88] == pytest.approx(wa * 0.5 / (wa + wv))
    assert pdf[89] == pytest.approx((wa * 0.5 + wv * 1.0) / (wa + wv))


def test_path_high_pdf_empty_when_models_lack_it_or_failed():
    assert alg._get_path_high_pdf({"accuweather_3h": {"temps": [88], "probs": [1.0]}}) == {}
    assert alg._get_path_high_pdf({"accuweather_3h": {"error": "boom"}, "var_3h": {"error": "boom"}}) == {}


def test_high_cannot_be_below_the_observed_high():
    pdf = alg._compute_daily_high_pdf({}, 90.2, {86: 0.2, 90: 0.5, 92: 0.3})
    assert min(pdf) == 90                       # 86 mass folded onto the rounded observed high
    assert pdf[90] == pytest.approx(0.7)
    assert pdf[92] == pytest.approx(0.3)
    assert sum(pdf.values()) == pytest.approx(1.0)


def test_falls_back_to_cdf_product_without_path_pdf():
    horizon = {"1h": {88: 0.5, 89: 0.5}, "2h": {88: 0.5, 89: 0.5}}
    pdf = alg._compute_daily_high_pdf(horizon, 88.0)
    # independent draws: P(max<=88)=0.25 -> the independence assumption is what makes this warm-biased
    assert pdf[88] == pytest.approx(0.25)
    assert pdf[89] == pytest.approx(0.75)
