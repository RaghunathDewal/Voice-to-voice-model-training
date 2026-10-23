"""(vision/prep_lvis.py) Filter Voxel51/LVIS (35k COCO images, 1,106 LVIS classes) down to the hotel-room classes and write
Qwen3-VL training rows (ms-swift / LLaMA-Factory "messages + images" format).

    python prep_lvis.py --out data/lvis --lvis-json lvis_v1_train.json --max-images 3000 --download
    python prep_lvis.py --out data/lvis --groups bedroom bathroom          # only some areas

Writes to --out:
    manifest.jsonl    one line per kept image: path, size-free boxes per YOUR class (Qwen 0-1000 scale)
    count_sft.jsonl   counting rows: "How many X?" -> {"objects":[{"label":..,"detected_count":..}]}
    ground_sft.jsonl  grounding rows: "Find every X" -> [{"bbox_2d":[x1,y1,x2,y2],"label":..}]
    stats.json        images / boxes per class, and how many count rows were skipped and why

THE MISSING-LABEL RULE (why your last fine-tune learned "always 0"):
  LVIS is federated: an image is NOT checked for every class. A class missing from an image's boxes means
  "unknown", not "0". So without --lvis-json this script only ever asks about classes that ARE boxed (count >= 1).
  With the official LVIS v1 json (https://www.lvisdataset.org/dataset) it also uses, per image:
    neg_category_ids             -> annotators verified the class is absent: a TRUE 0, safe to train on
    not_exhaustive_category_ids  -> not every instance was boxed: the count is only a lower bound, so skipped
"""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import os
import random
import time
from collections import Counter, defaultdict

import requests

REPO = "https://huggingface.co/datasets/Voxel51/LVIS/resolve/main"

# YOUR class  <-  exact LVIS label names, by area (image counts in this 35k copy in the comments).
# LVIS has NO door / window / stairs / floor / wall classes: take those from ADE20K or Open Images.
GROUPS = {
    "bedroom": {
        "pillow":         ["pillow", "cushion"],                              # 639 + 431: sofa cushions count as pillows
        "blanket":        ["blanket", "quilt", "bedspread"],                  # 596 + 138 + 31
        "bed":            ["bed", "bunk_bed", "crib"],                        # 615 + 7 + 14
        "headboard":      ["headboard"],                                      # 255
        "mattress":       ["mattress"],                                       # 77
        "lamp":           ["lamp", "table_lamp"],                             # 591 + 15: LVIS does not say WHERE it stands
        "lampshade":      ["lampshade"],                                      # 424
        "curtain":        ["curtain"],                                        # 626
        "alarm clock":    ["alarm_clock"],                                    # 53
        "wardrobe":       ["wardrobe", "cupboard"],                           # 1 + 91
        "drawer":         ["drawer"],                                         # 678
        "coat hanger":    ["coat_hanger"],                                    # 17
        "bathrobe":       ["bathrobe", "robe"],                               # 13 + 12
        "slippers":       ["slipper_(footwear)"],                             # 20
        "iron":           ["iron_(for_clothing)"],                            # 9
        "ironing board":  ["ironing_board"],                                  # 5
        "suitcase":       ["suitcase"],                                       # 550
    },
    "bathroom": {
        "towel":          ["towel", "bath_towel", "hand_towel"],              # 191 + 102 + 75
        "towel rack":     ["towel_rack"],                                     # 196
        "toilet":         ["toilet"],                                         # 656
        "toilet paper":   ["toilet_tissue"],                                  # 327
        "tissue box":     ["tissue_paper"],                                   # 102
        "sink":           ["sink"],                                           # 564 (kitchen_sink is under kitchen)
        "faucet":         ["faucet", "water_faucet"],                         # 669 + 24
        "bathtub":        ["bathtub"],                                        # 253
        "shower head":    ["shower_head"],                                    # 135
        "shower curtain": ["shower_curtain"],                                 # 135
        "bath mat":       ["bath_mat"],                                       # 86
        "mirror":         ["mirror"],                                         # 671 (rearview_mirror excluded)
        "soap":           ["soap"],                                           # 175
        "shampoo":        ["shampoo"],                                        # 35
        "dispenser":      ["dispenser"],                                      # 78 (soap / sanitiser)
        "toothbrush":     ["toothbrush"],                                     # 248
        "hairbrush":      ["hairbrush"],                                      # 41
        "hair dryer":     ["hair_dryer"],                                     # 45
    },
    "living": {
        "chair":          ["chair", "armchair", "folding_chair", "rocking_chair"],  # 643 + 195 + 22 + 17
        "sofa":           ["sofa"],                                           # 632
        "stool":          ["stool", "footstool", "ottoman", "step_stool"],    # 114 + 14 + 42 + 10
        "table":          ["table", "coffee_table", "dining_table", "kitchen_table"],  # 659 + 212 + 76 + 3
        "desk":           ["desk"],                                           # 357
        "bookcase":       ["bookcase"],                                       # 17
        "television":     ["television_set"],                                 # 641
        "remote control": ["remote_control"],                                 # 383
        "telephone":      ["telephone"],                                      # 244 (cellular_telephone excluded)
        "laptop":         ["laptop_computer"],                                # 613
        "monitor":        ["monitor_(computer_equipment) computer_monitor"],  # 446
        "speaker":        ["speaker_(stero_equipment)"],                      # 333
        "clock":          ["clock", "wall_clock"],                            # 637 + 21
        "painting":       ["painting", "poster"],                             # 362 + 295
        "vase":           ["vase"],                                           # 658
        "flowers":        ["flower_arrangement"],                             # 638
        "plant pot":      ["flowerpot"],                                      # 491
        "candle":         ["candle", "candle_holder"],                        # 379 + 65
        "book":           ["book", "magazine", "newspaper"],                  # 655 + 118 + 153
        "trash can":      ["trash_can"],                                      # 669
        "basket":         ["basket"],                                         # 582
        "fan":            ["fan"],                                            # 216
        "ashtray":        ["ashtray"],                                        # 31
    },
    "kitchen": {
        "refrigerator":   ["refrigerator"],                                   # 500
        "microwave":      ["microwave_oven"],                                 # 394
        "oven":           ["oven", "toaster_oven"],                           # 254 + 42
        "stove":          ["stove"],                                          # 355
        "dishwasher":     ["dishwasher"],                                     # 129
        "kitchen sink":   ["kitchen_sink"],                                   # 180
        "kettle":         ["kettle", "teakettle"],                            # 34 + 12
        "coffee maker":   ["coffee_maker", "coffeepot"],                      # 84 + 8
        "teapot":         ["teapot"],                                         # 46
        "toaster":        ["toaster"],                                        # 78
        "blender":        ["blender"],                                        # 87
        "cooking pot":    ["pot", "crock_pot", "saucepan"],                   # 136 + 15 + 2
        "frying pan":     ["frying_pan", "pan_(for_cooking)", "wok"],         # 39 + 58 + 2
        "cutting board":  ["chopping_board"],                                 # 225
        "plate":          ["plate", "paper_plate"],                           # 686 + 116
        "bowl":           ["bowl"],                                           # 658
        "mug":            ["mug", "cup", "teacup"],                           # 296 + 533 + 13
        "glass":          ["glass_(drink_container)", "wineglass", "flute_glass"],  # 652 + 330 + 9
        "spoon":          ["spoon", "soupspoon", "wooden_spoon"],             # 389 + 6 + 16
        "fork":           ["fork"],                                           # 680
        "knife":          ["knife"],                                          # 683 (pocketknife excluded)
        "chopsticks":     ["chopstick"],                                      # 51
        "ladle":          ["ladle"],                                          # 33
        "spatula":        ["spatula"],                                        # 100
        "tray":           ["tray"],                                           # 321
        "pitcher":        ["pitcher_(vessel_for_liquid)"],                    # 98
        "jar":            ["jar"],                                            # 150
        "water bottle":   ["water_bottle"],                                   # 226
        "wine bottle":    ["wine_bottle"],                                    # 184
        "bottle":         ["bottle"],                                         # 641
        "napkin":         ["napkin"],                                         # 663
        "tablecloth":     ["tablecloth"],                                     # 555
        "place mat":      ["place_mat"],                                      # 201
        "dish towel":     ["dishtowel"],                                      # 50
        "paper towel":    ["paper_towel"],                                    # 170
        "salt shaker":    ["saltshaker", "pepper_mill"],                      # 125 + 26
        "cooler":         ["cooler_(for_food)"],                              # 93
    },
    "building": {
        "doorknob":       ["doorknob"],                                       # 578
        "door mat":       ["doormat"],                                        # 22
        "wall socket":    ["wall_socket"],                                    # 688
        "light bulb":     ["lightbulb"],                                      # 310
        "chandelier":     ["chandelier"],                                     # 93
        "spotlight":      ["spotlight"],                                      # 20
        "air vent":       ["vent"],                                           # 515
        "air conditioner": ["air_conditioner"],                               # 119
        "radiator":       ["radiator", "heater"],                             # 61 + 19
        "thermostat":     ["thermostat"],                                     # 51
        "fire extinguisher": ["fire_extinguisher"],                           # 50
        "cabinet":        ["cabinet", "file_cabinet"],                        # 573 + 14
        "washing machine": ["automatic_washer"],                              # 18
        "bucket":         ["bucket"],                                         # 232
        "broom":          ["broom", "mop", "dustpan"],                        # 36 + 8 + 2
        "vacuum cleaner": ["vacuum_cleaner"],                                 # 10
        "bench":          ["bench"],                                          # 691
        "sun lounger":    ["deck_chair"],                                     # 80
        "umbrella":       ["umbrella"],                                       # 661
        "luggage cart":   ["handcart", "cart"],                               # 27 + 9
        "signboard":      ["signboard"],                                      # 643
        "mailbox":        ["mailbox_(at_home)"],                              # 41
        "wheelchair":     ["wheelchair"],                                     # 29
        "printer":        ["printer"],                                        # 57
    },
}
CLASS_MAP = {mine: lvs for g in GROUPS.values() for mine, lvs in g.items()}
LVIS_TO_MINE = {lv: mine for mine, lvs in CLASS_MAP.items() for lv in lvs}

COUNT_PROMPT = ("Count the following objects in this image: {names}. "
                'Answer in JSON: {{"objects":[{{"label":...,"detected_count":...}}]}}')
GROUND_PROMPT = 'Locate every {name} in the image. Answer in JSON: [{{"bbox_2d":[x1,y1,x2,y2],"label":"{name}"}}]'


def stream_samples(url: str, limit: int = 0):
    """Yield the samples of FiftyOne's {"samples": [ {...}, {...} ]} one by one, without loading 955 MB."""
    dec = json.JSONDecoder()
    buf, started, n = "", False, 0
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        for chunk in r.iter_content(1 << 20, decode_unicode=False):
            buf += chunk.decode("utf-8", errors="ignore")
            if not started:
                i = buf.find("[")
                if i < 0:
                    continue
                buf, started = buf[i + 1:], True
            while True:
                buf = buf.lstrip(" \n\r\t,")
                if not buf or buf[0] == "]":
                    break
                try:
                    obj, end = dec.raw_decode(buf)
                except json.JSONDecodeError:
                    break                     # sample not complete yet: read more
                buf = buf[end:]
                yield obj
                n += 1
                if limit and n >= limit:
                    return


def load_lvis_json(paths: list[str]) -> dict[int, tuple[set[str], set[str]]]:
    """COCO image id -> (verified-absent LVIS names, not-exhaustive LVIS names)."""
    out = {}
    for p in paths:
        d = json.load(open(p))
        names = {c["id"]: c["name"] for c in d["categories"]}
        for im in d["images"]:
            out[im["id"]] = ({names[c] for c in im.get("neg_category_ids", [])},
                             {names[c] for c in im.get("not_exhaustive_category_ids", [])})
    return out


def to_qwen_box(b: list[float]) -> list[int]:
    x, y, w, h = b                                   # FiftyOne: normalised [x, y, w, h]
    return [round(1000 * x), round(1000 * y), round(1000 * min(1, x + w)), round(1000 * min(1, y + h))]


def count_row(img: str, counts: dict) -> dict:
    names = list(counts)
    random.shuffle(names)
    answer = {"objects": [{"label": n, "detected_count": counts[n]} for n in names]}
    return {"messages": [{"role": "user", "content": "<image>" + COUNT_PROMPT.format(names=", ".join(names))},
                         {"role": "assistant", "content": json.dumps(answer)}], "images": [img]}


def ground_row(img: str, name: str, boxes: list) -> dict:
    return {"messages": [{"role": "user", "content": "<image>" + GROUND_PROMPT.format(name=name)},
                         {"role": "assistant", "content": json.dumps([{"bbox_2d": b, "label": name} for b in boxes])}],
            "images": [img]}


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="data/lvis")
    p.add_argument("--groups", nargs="*", default=None, help=f"areas to keep: {' '.join(GROUPS)} (default: all)")
    p.add_argument("--classes", nargs="*", default=None, help="subset of YOUR class names (overrides --groups)")
    p.add_argument("--priority", nargs="*", default=["pillow", "towel", "lamp", "chair"],
                   help="images with these classes are kept first when --max-images cuts")
    p.add_argument("--lvis-json", nargs="*", default=[], help="official lvis_v1_train.json (gives verified zeros)")
    p.add_argument("--max-images", type=int, default=0, help="keep at most N images (0 = all ~30k)")
    p.add_argument("--min-area", type=float, default=0.0005, help="boxes under this share of the image = too small")
    p.add_argument("--max-count", type=int, default=30, help="skip a count above this (too crowded to count)")
    p.add_argument("--max-zeros", type=int, default=2, help="verified-zero classes added per count row")
    p.add_argument("--max-names", type=int, default=8, help="classes asked about in one count row")
    p.add_argument("--download", action="store_true", help="download the kept images into OUT/images/")
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--limit", type=int, default=0, help="stop after N samples (testing)")
    p.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    random.seed(a.seed)
    wanted = set(a.classes or [c for g in (a.groups or GROUPS) for c in GROUPS[g]])
    unknown = wanted - set(CLASS_MAP)
    if unknown:
        raise SystemExit(f"not in CLASS_MAP: {sorted(unknown)}")
    lvis_info = load_lvis_json(a.lvis_json) if a.lvis_json else {}
    if not lvis_info:
        print("no --lvis-json: only classes that are boxed get asked (no zeros at all)")

    # ---- 1. stream the 35k samples, keep what we can use
    cands = []
    for s in stream_samples(f"{REPO}/samples.json", a.limit):
        boxes, small = defaultdict(list), Counter()
        for d in (s.get("detections") or {}).get("detections", []):
            mine = LVIS_TO_MINE.get(d["label"])
            if mine not in wanted:
                continue
            x, y, w, h = d["bounding_box"]
            if w * h < a.min_area:
                small[mine] += 1               # too small to see at Qwen's resolution, but it IS there
            else:
                boxes[mine].append(to_qwen_box(d["bounding_box"]))
        image_id = int(os.path.splitext(os.path.basename(s["filepath"]))[0])
        neg, not_exh = lvis_info.get(image_id, (set(), set()))
        neg_mine = {LVIS_TO_MINE[n] for n in neg if LVIS_TO_MINE.get(n) in wanted} - set(boxes) - set(small)
        not_exh_mine = {LVIS_TO_MINE[n] for n in not_exh if n in LVIS_TO_MINE}
        if boxes or neg_mine:
            cands.append({"filepath": s["filepath"], "coco_id": image_id, "boxes": dict(boxes), "small": dict(small),
                          "verified_absent": sorted(neg_mine), "not_exhaustive": sorted(not_exh_mine)})
    print(f"{len(cands)} usable images")

    # ---- 2. cut to --max-images, images with the priority classes first
    random.shuffle(cands)
    if a.max_images and len(cands) > a.max_images:
        pri = set(a.priority)
        cands.sort(key=lambda c: not (pri & set(c["boxes"])))
        cands = cands[:a.max_images]

    # ---- 3. download (only the kept images)
    os.makedirs(f"{a.out}/images", exist_ok=True)

    headers = {"Authorization": f"Bearer {os.environ['HF_TOKEN']}"} if os.environ.get("HF_TOKEN") else {}
    done = Counter()

    def fetch(c):
        local = os.path.abspath(f"{a.out}/images/{os.path.basename(c['filepath'])}")
        if not a.download:
            return f"{REPO}/{c['filepath']}"
        for attempt in range(5):
            if os.path.exists(local):
                break
            try:
                r = requests.get(f"{REPO}/{c['filepath']}", headers=headers, timeout=60)
                if r.status_code == 429:                 # rate limited: back off
                    time.sleep(10 * (attempt + 1))
                    continue
                r.raise_for_status()
                open(local + ".part", "wb").write(r.content)
                os.replace(local + ".part", local)
            except Exception as e:  # noqa: BLE001
                if attempt == 4:
                    print("download failed:", c["filepath"], e, flush=True)
                time.sleep(2 * (attempt + 1))
        done["n"] += 1
        if done["n"] % 2000 == 0:
            print(f"  downloaded {done['n']}/{len(cands)}", flush=True)
        return local if os.path.exists(local) else None
    with ThreadPoolExecutor(a.workers) as ex:
        paths = list(ex.map(fetch, cands))

    # ---- 4. rows
    st = defaultdict(Counter)
    f_man = open(f"{a.out}/manifest.jsonl", "w")
    f_cnt = open(f"{a.out}/count_sft.jsonl", "w")
    f_gnd = open(f"{a.out}/ground_sft.jsonl", "w")
    for c, img in zip(cands, paths):
        if img is None:
            continue
        st["all"]["images"] += 1
        f_man.write(json.dumps({"image": img, **{k: v for k, v in c.items() if k != "filepath"}}) + "\n")
        counts = {}
        for name, bs in c["boxes"].items():
            st[name]["images"] += 1
            st[name]["boxes"] += len(bs)
            if name in c["not_exhaustive"]:
                st[name]["skip_not_exhaustive"] += 1   # count is only a lower bound: do not teach it
            elif c["small"].get(name):
                st[name]["skip_has_tiny_boxes"] += 1   # tiny ones dropped -> the box count is not the truth
            elif len(bs) > a.max_count:
                st[name]["skip_too_many"] += 1
            else:
                counts[name] = len(bs)
                f_gnd.write(json.dumps(ground_row(img, name, bs)) + "\n")
                st["all"]["ground_rows"] += 1
        present = list(counts)
        if len(present) > a.max_names - 1:            # leave room for at least one zero
            present = random.sample(present, a.max_names - 1)
        counts = {n: counts[n] for n in present}
        zeros = c["verified_absent"][:]
        random.shuffle(zeros)
        zeros.sort(key=lambda z: z not in a.priority)  # true zeros of the main classes first
        for z in zeros[:min(a.max_zeros, a.max_names - len(counts))]:
            counts[z] = 0                              # verified absent (only with --lvis-json)
            st[z]["zero_rows"] += 1
        if counts:
            f_cnt.write(json.dumps(count_row(img, counts)) + "\n")
            st["all"]["count_rows"] += 1
    json.dump({k: dict(v) for k, v in sorted(st.items())}, open(f"{a.out}/stats.json", "w"), indent=1)
    print(json.dumps(dict(st["all"])))
    for name in sorted(a.priority):
        print(f"  {name:10s} {dict(st[name])}")


if __name__ == "__main__":
    main()
