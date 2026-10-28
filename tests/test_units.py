"""Units of the energy / carbon chain on a hand-made job (week 32 s_C audit): requests in machines (1.0 = the
largest machine, declared 64 cores / 256 GB), duration in hours, power in kW, energy in kWh, carbon intensity in
kgCO2/kWh, carbon in kg, carbon cost = s_C [$/kg] x kg, electricity cost = kWh x $/kWh."""
import numpy as np
import pytest

from dlent_rt.energy import EnergyParams, compute_energy_kwh, compute_power_kw
from dlent_rt.reward import carbon_kg, carbon_price, electricity_cost


def test_energy_and_carbon_units_hand_made_job():
    p = EnergyParams(machine_cores=64, machine_ram_gb=256, p_core=15.0, p_gb=0.35, pue=1.10)
    cpu, ram, sc, hours = np.array([0.5]), np.array([0.25]), np.array([0.0]), np.array([2.0])
    # 0.5 machine = 32 cores x 15 W = 480 W; 0.25 machine = 64 GB x 0.35 W = 22.4 W; x PUE 1.10 -> 0.55264 kW
    kw = compute_power_kw(cpu, ram, sc, p)
    assert kw[0] == pytest.approx(1.10 * (32 * 15.0 + 64 * 0.35) / 1000)
    kwh = compute_energy_kwh(cpu, ram, sc, hours, p)
    assert kwh[0] == pytest.approx(kw[0] * 2.0)                       # 1.10528 kWh in 2 hours
    assert carbon_kg(kwh, np.array([0.42]))[0] == pytest.approx(0.42 * kwh[0])        # kg = kWh x kg/kWh
    assert electricity_cost(kwh, np.array([0.025]))[0] == pytest.approx(0.025 * kwh[0])   # $ = kWh x $/kWh


def test_carbon_price_is_a_ratio_of_quantiles():
    # s_C = D_q / K_q: the map sends the q-quantile of carbon (kg) onto the q-quantile of the dollar terms ($)
    cv, pi, K, p = np.array([10.0, 20.0]), np.array([2.0, -4.0]), np.array([0.001, 0.002]), np.array([0.5, 0.5])
    out = carbon_price(cv, pi, K, p, 0.9)
    assert out["s_C"] == pytest.approx(0.5 * (out["Q_cv"] + out["Q_abs_pi"]) / out["K_q"])
