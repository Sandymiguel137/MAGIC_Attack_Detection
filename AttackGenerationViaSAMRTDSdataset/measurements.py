import numpy as np
import opendssdirect as dss
from scipy.sparse import csc_matrix

def get_all_node_voltageReIm_map_afterPF():
    dss_map = {}
    for bus_name in dss.Circuit.AllBusNames():
        dss.Circuit.SetActiveBus(bus_name)
        voltages = dss.Bus.PuVoltage()
        phases = dss.Bus.Nodes()
        for i, ph in enumerate(phases):
            key = f"{bus_name.lower()}.{ph}"
            dss_map[key] = (voltages[2 * i], voltages[2 * i + 1])
    return dss_map

def get_per_unit_current_map_from_Ybus():
    from scipy.sparse import csc_matrix
    Y_sparse = csc_matrix(dss.YMatrix.getYsparse()) 
    Y_NodeOrder = dss.Circuit.YNodeOrder()
    voltage_map = get_all_node_voltageReIm_map_afterPF()
    N = len(Y_NodeOrder)

    V = np.zeros(N, dtype=complex)
    for i, node in enumerate(Y_NodeOrder):
        key = node.lower()
        if key in voltage_map:
            re_raw, im_raw = voltage_map[key]
            re = float(np.asarray(re_raw).item())
            im = float(np.asarray(im_raw).item())
            V[i] = complex(re, im)
        else:
            raise KeyError(f"Voltage for node '{key}' not found in voltage_map.")

    I = Y_sparse @ V

    current_map = {}
    for i, node in enumerate(Y_NodeOrder):
        current_map[node.lower()] = (I[i].real, I[i].imag)

    return current_map


def measurements(S_mu_PMU, S_power_meter, sigma_v=0.01, scale_i=0.01):
    # Ensure disjoint sensor sets
    set_mu = set(x.lower() for x in S_mu_PMU)
    set_pm = set(x.lower() for x in S_power_meter)
    if set_mu & set_pm:
        raise ValueError(f"Sensor sets must be disjoint. Overlap: {set_mu & set_pm}")

    voltage_map = get_all_node_voltageReIm_map_afterPF()
    current_map = get_per_unit_current_map_from_Ybus()

    # Adaptive current noise std: sigma_i = 0.5 * sqrt(mean(|i|^2)) * scale
    i_sq = [np.abs(complex(*i))**2 for i in current_map.values()]
    sigma_i = 0.5 * np.sqrt(np.mean(i_sq)) * scale_i

    def complex_noise(std):
        return np.random.normal(0, std) + 1j * np.random.normal(0, std)

    z_dict = {}
    matched_mu, matched_pm = [], []

    for sensor in S_mu_PMU:
        key = sensor.lower()
        if key in voltage_map and key in current_map:
            v = complex(*voltage_map[key])
            i = complex(*current_map[key])
            u_m = v + complex_noise(sigma_v)
            c_m = i + complex_noise(sigma_i)
            z_dict[key] = (u_m, c_m)
            matched_mu.append(key)

    for sensor in S_power_meter:
        key = sensor.lower()
        if key in voltage_map and key in current_map:
            v = complex(*voltage_map[key])
            i = complex(*current_map[key])
            u_m = v + complex_noise(sigma_v)
            c_m = i + complex_noise(sigma_i)
            s_m = u_m * np.conj(c_m)
            z_dict[key] = (s_m, abs(u_m), abs(c_m))
            matched_pm.append(key)

    print(f"mu-PMU: {len(matched_mu)}/{len(S_mu_PMU)} matched")
    if len(matched_mu) < len(S_mu_PMU):
        print("Unmatched mu-PMU examples:", list(set_mu - set(matched_mu))[:5])

    print(f"PowerMeter: {len(matched_pm)}/{len(S_power_meter)} matched")
    if len(matched_pm) < len(S_power_meter):
        print("Unmatched PowerMeter examples:", list(set_pm - set(matched_pm))[:5])

    return z_dict
