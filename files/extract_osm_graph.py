"""
AegisFlow — OSM road-graph extraction pipeline
================================================

Turns a raw OpenStreetMap XML extract into the same kind of topological
graph OSMnx produces: nodes = real intersections, edges = real road
segments carrying length, highway class, speed limit, lane count and
directionality.

This script has NO third-party dependencies (stdlib only) so it runs
anywhere. If you have real network access and want to pull a live city
instead of using the bundled extract, see `USING_REAL_OSMNX.md` — three
lines of OSMnx replace steps 1-2 below and everything downstream
(simplification, JSON export) is unaffected.

Pipeline:
  1. Parse <node> and <way> elements from the .osm XML.
  2. Keep only drivable highway classes.
  3. Compute real haversine segment lengths.
  4. Simplify the raw path graph to an intersections-only topological
     graph (this is exactly what osmnx.simplification.simplify_graph
     does: nodes of degree != 2, plus way endpoints, are "real"
     intersections; everything between them collapses into one edge).
  5. Attach traffic-engineering metadata (speed, lane count, capacity)
     using the OSM tag when present, else a standard default by road
     class — the normal practice in transportation modeling, since raw
     OSM data does not carry vehicle-throughput capacity directly.
  6. Export nodes/edges/roles to JSON for the browser simulation engine.

Usage:
    python3 extract_osm_graph.py West-Oakland.osm oakland_graph.json
"""

import sys
import json
import math
import collections
import xml.etree.ElementTree as ET

# ---------------------------------------------------------------------------
# Config: which highway classes count as "drivable", and the defaults used
# when a tag is missing. Numbers are standard-practice placeholders, not
# measured data — replace with your own model's assumptions as needed.
# ---------------------------------------------------------------------------
EXCLUDED_HIGHWAYS = {
    "footway", "path", "steps", "pedestrian", "cycleway",
    "traffic_signals", "stop",
}
DEFAULT_SPEED_KMH = {
    "motorway": 100, "trunk": 80, "primary": 60, "secondary": 45,
    "tertiary": 40, "residential": 40, "unclassified": 40,
    "service": 20, "track": 20,
}
CAPACITY_PER_LANE = {
    "motorway": 2200, "trunk": 1800, "primary": 1500, "secondary": 900,
    "tertiary": 700, "residential": 500, "unclassified": 500,
    "service": 250, "track": 250,
}


def haversine(a, b):
    """Great-circle distance in meters between (lat, lon) points a and b."""
    lat1, lon1 = a
    lat2, lon2 = b
    r = 6371000
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    h = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(h))


def parse_osm(path):
    """Step 1-2: read raw nodes and drivable ways out of the .osm XML."""
    tree = ET.parse(path)
    root = tree.getroot()

    nodes = {}
    for nd in root.findall("node"):
        nodes[nd.get("id")] = (float(nd.get("lat")), float(nd.get("lon")))

    ways = []
    for w in root.findall("way"):
        tags = {t.get("k"): t.get("v") for t in w.findall("tag")}
        hw = tags.get("highway")
        if not hw or hw in EXCLUDED_HIGHWAYS:
            continue
        refs = [nd.get("ref") for nd in w.findall("nd")]
        refs = [r for r in refs if r in nodes]
        if len(refs) < 2:
            continue
        ways.append({"id": w.get("id"), "refs": refs, "tags": tags})

    return nodes, ways


def simplify_to_topology(nodes, ways):
    """Step 3-4: collapse the raw node/way graph to an intersections-only
    topological graph, mirroring osmnx.simplification.simplify_graph."""
    deg = collections.Counter()
    endpoints = set()
    for w in ways:
        refs = w["refs"]
        endpoints.add(refs[0])
        endpoints.add(refs[-1])
        for i in range(len(refs) - 1):
            deg[refs[i]] += 1
            deg[refs[i + 1]] += 1
    # a node survives simplification if it's a real branch point/dead end
    # (degree != 2) or the endpoint of some way
    kept = {n for n, d in deg.items() if d != 2} | endpoints

    edges = []
    for w in ways:
        refs = w["refs"]
        seg_start = 0
        for i in range(1, len(refs)):
            if refs[i] in kept:
                sub = refs[seg_start:i + 1]
                length = sum(
                    haversine(nodes[sub[j]], nodes[sub[j + 1]])
                    for j in range(len(sub) - 1)
                )
                if length > 1:
                    edges.append({"u": sub[0], "v": sub[-1], "length_m": round(length, 1),
                                  "tags": w["tags"]})
                seg_start = i
    return edges


def largest_connected_component(edges):
    """Real OSM extracts often contain small disconnected fragments
    (parking lots, driveways clipped by the bounding box). Keep only the
    main routable component."""
    adj = collections.defaultdict(list)
    used = set()
    for e in edges:
        adj[e["u"]].append(e["v"])
        adj[e["v"]].append(e["u"])
        used.add(e["u"])
        used.add(e["v"])

    seen, comps = set(), []
    for n in used:
        if n in seen:
            continue
        stack, comp = [n], set()
        while stack:
            x = stack.pop()
            if x in comp:
                continue
            comp.add(x)
            seen.add(x)
            stack.extend(y for y in adj[x] if y not in comp)
        comps.append(comp)
    comps.sort(key=len, reverse=True)
    main = comps[0]
    return [e for e in edges if e["u"] in main and e["v"] in main], main


def attach_metadata(edges):
    """Step 5: real OSM tag where present, standard default by class
    otherwise. This is exactly the assumption most SIH-style traffic
    models make, since OSM carries no vehicle-throughput field."""
    out = []
    for e in edges:
        tags = e["tags"]
        hw = tags.get("highway", "unclassified")
        speed = int(tags["maxspeed"]) if tags.get("maxspeed", "").isdigit() else DEFAULT_SPEED_KMH.get(hw, 35)
        lanes = int(tags["lanes"]) if tags.get("lanes", "").isdigit() else (2 if hw in ("secondary", "primary", "trunk") else 1)
        capacity = lanes * CAPACITY_PER_LANE.get(hw, 400)
        freeflow_min = e["length_m"] / (speed * 1000 / 60) if speed else e["length_m"] / 500
        out.append({
            "u": e["u"], "v": e["v"], "length_m": e["length_m"],
            "highway": hw, "name": tags.get("name"),
            "speed_kmh": speed, "lanes": lanes, "capacity": round(capacity),
            "oneway": tags.get("oneway") == "yes",
            "freeflow_min": round(freeflow_min, 3),
        })
    return out


def main():
    if len(sys.argv) != 3:
        print("usage: extract_osm_graph.py <input.osm> <output.json>")
        sys.exit(1)
    osm_path, out_path = sys.argv[1], sys.argv[2]

    nodes, ways = parse_osm(osm_path)
    print(f"parsed {len(nodes)} raw nodes, {len(ways)} drivable ways")

    raw_edges = simplify_to_topology(nodes, ways)
    edges, main_component = largest_connected_component(raw_edges)
    print(f"simplified topology: {len(main_component)} intersections, {len(edges)} road segments")

    edges = attach_metadata(edges)
    out_nodes = {n: {"lat": nodes[n][0], "lon": nodes[n][1]} for n in main_component}

    with open(out_path, "w") as f:
        json.dump({"nodes": out_nodes, "edges": edges}, f, indent=2)
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
