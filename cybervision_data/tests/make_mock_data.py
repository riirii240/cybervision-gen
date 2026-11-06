"""Faux CIC-IDS2017 (format TrafficLabelling) reproduisant ses défauts connus, pour tester le pipeline
sans télécharger 1 Go : espaces dans les noms, colonne "Fwd Header Length.1", inf/NaN, négatifs,
doublons exacts, labels contradictoires, caractère illisible "Web Attack � Brute Force",
lignes vides, timestamps jj/mm/aaaa et après-midi sur 12 h, colonnes constantes.
"""
import os
import sys

import numpy as np
import pandas as pd

FEATS = ["Flow Duration", "Total Fwd Packets", "Total Backward Packets", "Total Length of Fwd Packets",
         "Total Length of Bwd Packets", "Fwd Packet Length Max", "Fwd Packet Length Min",
         "Fwd Packet Length Mean", "Fwd Packet Length Std", "Bwd Packet Length Max", "Bwd Packet Length Min",
         "Bwd Packet Length Mean", "Bwd Packet Length Std", "Flow Bytes/s", "Flow Packets/s", "Flow IAT Mean",
         "Flow IAT Std", "Flow IAT Max", "Flow IAT Min", "Fwd IAT Total", "Fwd IAT Mean", "Fwd IAT Std",
         "Fwd IAT Max", "Fwd IAT Min", "Bwd IAT Total", "Bwd IAT Mean", "Bwd IAT Std", "Bwd IAT Max",
         "Bwd IAT Min", "Fwd PSH Flags", "Bwd PSH Flags", "Fwd URG Flags", "Bwd URG Flags", "Fwd Header Length",
         "Bwd Header Length", "Fwd Packets/s", "Bwd Packets/s", "Min Packet Length", "Max Packet Length",
         "Packet Length Mean", "Packet Length Std", "Packet Length Variance", "FIN Flag Count", "SYN Flag Count",
         "RST Flag Count", "PSH Flag Count", "ACK Flag Count", "URG Flag Count", "CWE Flag Count",
         "ECE Flag Count", "Down/Up Ratio", "Average Packet Size", "Avg Fwd Segment Size",
         "Avg Bwd Segment Size", "Fwd Header Length.1", "Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk",
         "Fwd Avg Bulk Rate", "Bwd Avg Bytes/Bulk", "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate",
         "Subflow Fwd Packets", "Subflow Fwd Bytes", "Subflow Bwd Packets", "Subflow Bwd Bytes",
         "Init_Win_bytes_forward", "Init_Win_bytes_backward", "act_data_pkt_fwd", "min_seg_size_forward",
         "Active Mean", "Active Std", "Active Max", "Active Min", "Idle Mean", "Idle Std", "Idle Max", "Idle Min"]

# jour -> [(label, heure début, heure fin, n)]
PLAN = {
    "Monday": [("BENIGN", 9, 17, 3000)],
    "Tuesday": [("BENIGN", 9, 17, 2500), ("FTP-Patator", 9.3, 10.3, 1400), ("SSH-Patator", 14, 15, 1300)],
    "Wednesday": [("BENIGN", 9, 17, 2500), ("DoS slowloris", 9.7, 10.2, 1200), ("DoS Hulk", 10.7, 11.4, 1800),
                  ("Heartbleed", 15.2, 15.5, 11)],
    "Thursday-WorkingHours-Morning-WebAttacks": [("BENIGN", 9, 12, 2000),
                  ("Web Attack \x96 Brute Force", 9.3, 10, 900), ("Web Attack \x96 XSS", 10.25, 10.6, 400),
                  ("Web Attack \x96 Sql Injection", 10.65, 10.7, 21)],
    "Friday": [("BENIGN", 9, 17, 2500), ("Bot", 10, 11, 1100), ("PortScan", 13.9, 14.6, 1500),
               ("DDoS", 15.9, 16.3, 1500)],
}


def gen_flows(label, n, rng):
    k = abs(hash(label)) % 7 + 1
    nf = rng.poisson(2 + k, n) + 1
    nb = rng.poisson(1 + k / 2, n)
    lf_max = rng.integers(0, 1500, n).astype(float)
    lf_min = np.floor(lf_max * rng.uniform(0, 0.5, n))
    lf_mean = lf_min + (lf_max - lf_min) * rng.uniform(0, 1, n)
    lb_max = rng.integers(0, 3000, n).astype(float) * (nb > 0)
    lb_min = np.floor(lb_max * rng.uniform(0, 0.5, n))
    lb_mean = lb_min + (lb_max - lb_min) * rng.uniform(0, 1, n)
    dur = rng.lognormal(10 + k / 2, 2, n).round()
    iat_max = dur * rng.uniform(0.3, 1, n)
    iat_min = iat_max * rng.uniform(0, 0.2, n)
    iat_mean = iat_min + (iat_max - iat_min) * rng.uniform(0, 1, n)
    totf, totb = nf * lf_mean, nb * lb_mean
    d = {
        "Flow Duration": dur, "Total Fwd Packets": nf, "Total Backward Packets": nb,
        "Total Length of Fwd Packets": totf.round(), "Total Length of Bwd Packets": totb.round(),
        "Fwd Packet Length Max": lf_max, "Fwd Packet Length Min": lf_min, "Fwd Packet Length Mean": lf_mean,
        "Fwd Packet Length Std": (lf_max - lf_min) / 3, "Bwd Packet Length Max": lb_max,
        "Bwd Packet Length Min": lb_min, "Bwd Packet Length Mean": lb_mean, "Bwd Packet Length Std": (lb_max - lb_min) / 3,
        "Flow Bytes/s": (totf + totb) / (dur / 1e6 + 1e-9), "Flow Packets/s": (nf + nb) / (dur / 1e6 + 1e-9),
        "Flow IAT Mean": iat_mean, "Flow IAT Std": (iat_max - iat_min) / 4, "Flow IAT Max": iat_max, "Flow IAT Min": iat_min,
        "Fwd IAT Total": dur * rng.uniform(0.5, 1, n), "Bwd IAT Total": dur * rng.uniform(0, 0.9, n),
        "Fwd PSH Flags": rng.integers(0, 2, n), "Bwd PSH Flags": 0, "Fwd URG Flags": 0, "Bwd URG Flags": 0,
        "Fwd Header Length": nf * 32, "Bwd Header Length": nb * 32,
        "Fwd Packets/s": nf / (dur / 1e6 + 1e-9), "Bwd Packets/s": nb / (dur / 1e6 + 1e-9),
        "Min Packet Length": np.minimum(lf_min, np.where(nb > 0, lb_min, lf_min)),
        "Max Packet Length": np.maximum(lf_max, lb_max),
        "Packet Length Mean": (lf_mean + lb_mean) / 2,
    }
    d["Packet Length Std"] = (d["Max Packet Length"] - d["Min Packet Length"]) / 3
    d["Packet Length Variance"] = d["Packet Length Std"] ** 2
    for f in ["FIN", "SYN", "RST", "PSH", "ACK", "URG", "ECE"]:
        d[f"{f} Flag Count"] = rng.integers(0, 2, n)
    d["CWE Flag Count"] = 0
    d["Down/Up Ratio"] = np.floor(nb / nf)
    d["Average Packet Size"] = (totf + totb) / (nf + nb)
    d["Avg Fwd Segment Size"], d["Avg Bwd Segment Size"] = lf_mean, lb_mean
    d["Fwd Header Length.1"] = d["Fwd Header Length"]
    for c in ["Fwd Avg Bytes/Bulk", "Fwd Avg Packets/Bulk", "Fwd Avg Bulk Rate", "Bwd Avg Bytes/Bulk",
              "Bwd Avg Packets/Bulk", "Bwd Avg Bulk Rate"]:
        d[c] = 0
    d["Subflow Fwd Packets"], d["Subflow Fwd Bytes"] = nf, d["Total Length of Fwd Packets"]
    d["Subflow Bwd Packets"], d["Subflow Bwd Bytes"] = nb, d["Total Length of Bwd Packets"]
    d["Init_Win_bytes_forward"] = np.where(rng.uniform(size=n) < 0.2, -1, rng.integers(0, 65535, n))
    d["Init_Win_bytes_backward"] = np.where(rng.uniform(size=n) < 0.4, -1, rng.integers(0, 65535, n))
    d["act_data_pkt_fwd"] = np.floor(nf * rng.uniform(0, 1, n))
    d["min_seg_size_forward"] = rng.choice([20, 32], n)
    for g in ["Fwd IAT", "Bwd IAT", "Active", "Idle"]:
        base = d.get(f"{g} Total", dur * rng.uniform(0, 0.5, n))
        mx = base * rng.uniform(0.2, 1, n)
        mn = mx * rng.uniform(0, 0.3, n)
        d[f"{g} Max"], d[f"{g} Min"] = mx, mn
        d[f"{g} Mean"] = mn + (mx - mn) * rng.uniform(0, 1, n)
        d[f"{g} Std"] = (mx - mn) / 4
    return pd.DataFrame({c: d[c] if np.ndim(d[c]) else np.full(n, d[c]) for c in FEATS})


def fmt_ts(day_idx, hours, rng):
    """Format original : 'd/7/2017 h:mm' ; après-midi écrit sur 12 h (sans PM)."""
    out = []
    for h in hours:
        hh, mm = int(h), int((h - int(h)) * 60)
        hh12 = hh - 12 if hh > 12 else hh
        out.append(f"{3 + day_idx}/7/2017 {hh12}:{mm:02d}")
    return out


def make(out_dir, seed=0):
    rng = np.random.default_rng(seed)
    os.makedirs(out_dir, exist_ok=True)
    for di, (day, plan) in enumerate(PLAN.items()):
        parts = []
        for label, h0, h1, n in plan:
            f = gen_flows(label, n, rng)
            hours = np.sort(rng.uniform(h0, h1, n))
            f.insert(0, "Timestamp", fmt_ts(di, hours, rng))
            f.insert(0, "Protocol", 6)
            f.insert(0, "Destination Port", rng.choice([80, 21, 22, 443], n))
            f.insert(0, "Destination IP", "192.168.10.50")
            f.insert(0, "Source Port", rng.integers(1024, 65535, n))
            f.insert(0, "Source IP", "172.16.0.1")
            f.insert(0, "Flow ID", "x")
            f["Label"] = label
            f["_h"] = hours
            parts.append(f)
        df = pd.concat(parts).sort_values("_h").drop(columns="_h").reset_index(drop=True)
        # défauts connus
        idx = rng.choice(len(df), 30, replace=False)
        df.loc[idx[:10], "Flow Bytes/s"] = np.inf
        df.loc[idx[10:15], "Flow Bytes/s"] = np.nan
        df.loc[idx[15:20], "Flow IAT Min"] = -3
        dups = df.sample(200, random_state=seed + di)
        conflict = df[df["Label"] == "BENIGN"].sample(5, random_state=seed + di).assign(Label="DoS Hulk" if day == "Wednesday" else "Bot")
        df = pd.concat([df, dups, conflict]).reset_index(drop=True)
        df.columns = [" " + c if i % 2 else c for i, c in enumerate(df.columns)]  # espaces parasites
        if "WebAttacks" in day:
            df = pd.concat([df, pd.DataFrame(np.nan, index=range(50), columns=df.columns)])
        name = day if "-" in day else f"{day}-WorkingHours"
        df.to_csv(os.path.join(out_dir, f"{name}.pcap_ISCX.csv"), index=False, encoding="latin-1")


if __name__ == "__main__":
    make(sys.argv[1] if len(sys.argv) > 1 else "data/raw")
