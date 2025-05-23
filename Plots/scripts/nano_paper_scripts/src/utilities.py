
def convert_eff_to_rate(eff, nBunches = 2544):
    return eff * (float(nBunches) * 11425e-3)

def convert_rate_to_eff(rate, nBunches = 2544):
    return rate / (float(nBunches) * 11425e-3)
