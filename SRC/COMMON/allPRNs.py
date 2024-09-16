
def allprns():
    all_const = {}
    # Fill dict with Galileo PRNs (E01 to E36)
    for prn_gal in range(1, 37):
        if prn_gal <= 9:
            all_const[f"E0{prn_gal}"] = prn_gal
        else:
            all_const[f"E{prn_gal}"] = prn_gal

    # Fill dict with GPS PRNs (G01 to G36)
    for prn_gps in range(1, 37):
        if prn_gps <= 9:
            all_const[f"G0{prn_gps}"] = prn_gps + prn_gal  # Adjusting the range to avoid overlap with Galileo
        else:
            all_const[f"G{prn_gps}"] = prn_gps + prn_gal

    return all_const