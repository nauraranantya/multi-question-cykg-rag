#!/usr/bin/env python3
"""Build the local MITRE ATT&CK knowledge graph the MCP server loads with --rdf-file.

Converts an official ATT&CK STIX 2.1 bundle (e.g. enterprise-attack-16.1.json from
https://github.com/mitre-attack/attack-stix-data) into Turtle in the shape of the
original SEPSES ATT&CK graph (http://w3id.org/sepses/vocab/ref/attack#), so every
query in src/mcp-cskg-rdf/server.py works unchanged:

  - technique/sub-technique IRIs end in the ATT&CK ID (.../attack/technique/T1505.003),
    which get_technique_by_id and get_technique_mitigations_by_keyword rely on;
    sub-techniques are typed both attack:Technique and attack:SubTechnique
  - attack:hasMitigation / attack:preventsTechnique, attack:accomplishesTactic,
    attack:isSubTechniqueOf, attack:platform
  - groups: attack:usesTechnique, attack:usesSoftware (tools), attack:usesMalware, attack:aliases
  - software: technique attack:hasSoftware tool, malware attack:implementsTechnique technique
    (the directions the server's union query expects), emitted for tools and malware alike

Why local: the public SEPSES endpoint the server used (w3id.org/sepses/sparql ->
sepses.ifs.tuwien.ac.at) stopped answering in October 2026, and the new SEPSES-KG
(sepses.org) holds ATT&CK techniques only, without IDs, tactics or mitigations.
A pinned release also makes the knowledge graph match the ATT&CK version the
evaluation references are adjudicated against.

Revoked and deprecated objects (and relationships touching them) are left out, so
e.g. the legacy per-technique "... Mitigation" courses of action do not appear.
Campaigns and data-component "detects" links have no counterpart in the server's
schema and are skipped. "(Citation: ...)" markers are stripped from descriptions.

Usage:
  python mcp-cskg-rdf/build_attack_rdf.py eval/reference_sources/enterprise-attack-16.1.json \
      mcp-cskg-rdf/data/attack-enterprise-16.1.ttl
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

from rdflib import Graph, Literal, Namespace, URIRef
from rdflib.namespace import DCTERMS, RDF, XSD

ATTACK = Namespace("http://w3id.org/sepses/vocab/ref/attack#")
RES = "http://w3id.org/sepses/resource/attack/"
_CITATION = re.compile(r"\s*\(Citation:[^)]*\)")


def _external_id(obj: dict) -> str | None:
    for ref in obj.get("external_references", []):
        if ref.get("source_name") == "mitre-attack" and ref.get("external_id"):
            return ref["external_id"]
    return None


def _active(obj: dict) -> bool:
    return not obj.get("revoked") and not obj.get("x_mitre_deprecated")


def _clean(text: str | None) -> str | None:
    if not text:
        return None
    return _CITATION.sub("", text).strip()


def build(bundle_path: Path) -> tuple[Graph, dict]:
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    objects = [o for o in bundle["objects"] if _active(o)]
    by_id = {o["id"]: o for o in objects}

    g = Graph()
    g.bind("attack", ATTACK)
    g.bind("dcterm", DCTERMS)

    iri: dict[str, URIRef] = {}
    kind_path = {
        "attack-pattern": "technique", "x-mitre-tactic": "tactic", "course-of-action": "mitigation",
        "intrusion-set": "group", "malware": "software", "tool": "software",
        "x-mitre-data-source": "data-source", "x-mitre-data-component": "data-component",
    }
    kind_class = {
        "x-mitre-tactic": ATTACK.Tactic, "course-of-action": ATTACK.Mitigation,
        "intrusion-set": ATTACK.AdversaryGroup, "malware": ATTACK.Malware, "tool": ATTACK.Software,
        "x-mitre-data-source": ATTACK.DataSource, "x-mitre-data-component": ATTACK.DataComponent,
    }

    for o in objects:
        t = o["type"]
        if t not in kind_path:
            continue
        ext = _external_id(o)
        local = ext or o["id"].split("--", 1)[1]
        node = URIRef(f"{RES}{kind_path[t]}/{local}")
        iri[o["id"]] = node
        if t == "attack-pattern":
            g.add((node, RDF.type, ATTACK.Technique))
            if o.get("x_mitre_is_subtechnique"):
                g.add((node, RDF.type, ATTACK.SubTechnique))
            for platform in o.get("x_mitre_platforms", []):
                g.add((node, ATTACK.platform, Literal(platform)))
        else:
            g.add((node, RDF.type, kind_class[t]))
        g.add((node, DCTERMS.title, Literal(o["name"])))
        if ext:
            g.add((node, DCTERMS.identifier, Literal(ext)))
        description = _clean(o.get("description"))
        if description:
            g.add((node, DCTERMS.description, Literal(description)))
        for stamp, prop in (("created", DCTERMS.created), ("modified", DCTERMS.modified)):
            if o.get(stamp):
                g.add((node, prop, Literal(o[stamp], datatype=XSD.dateTime)))
        for alias in o.get("aliases", []) + o.get("x_mitre_aliases", []):
            if alias != o["name"]:
                g.add((node, ATTACK.aliases, Literal(alias)))

    # technique -> tactic, via kill-chain phase short names
    tactic_by_short = {o["x_mitre_shortname"]: iri[o["id"]] for o in objects
                       if o["type"] == "x-mitre-tactic" and o["id"] in iri}
    for o in objects:
        if o["type"] == "attack-pattern" and o["id"] in iri:
            for phase in o.get("kill_chain_phases", []):
                if phase.get("kill_chain_name") == "mitre-attack" and phase["phase_name"] in tactic_by_short:
                    g.add((iri[o["id"]], ATTACK.accomplishesTactic, tactic_by_short[phase["phase_name"]]))

    counts = {"mitigates": 0, "subtechnique-of": 0, "group-technique": 0, "group-software": 0,
              "software-technique": 0}
    for r in objects:
        if r["type"] != "relationship":
            continue
        src, dst = by_id.get(r["source_ref"]), by_id.get(r["target_ref"])
        if src is None or dst is None or r["source_ref"] not in iri or r["target_ref"] not in iri:
            continue   # an endpoint is revoked/deprecated or has no counterpart (campaign)
        s, d, rel = iri[r["source_ref"]], iri[r["target_ref"]], r["relationship_type"]
        st, dt = src["type"], dst["type"]
        if rel == "mitigates" and st == "course-of-action" and dt == "attack-pattern":
            g.add((s, ATTACK.preventsTechnique, d))
            g.add((d, ATTACK.hasMitigation, s))
            counts["mitigates"] += 1
        elif rel == "subtechnique-of" and st == dt == "attack-pattern":
            g.add((s, ATTACK.isSubTechniqueOf, d))
            counts["subtechnique-of"] += 1
        elif rel == "uses" and st == "intrusion-set" and dt == "attack-pattern":
            g.add((s, ATTACK.usesTechnique, d))
            counts["group-technique"] += 1
        elif rel == "uses" and st == "intrusion-set" and dt in ("tool", "malware"):
            g.add((s, ATTACK.usesSoftware if dt == "tool" else ATTACK.usesMalware, d))
            counts["group-software"] += 1
        elif rel == "uses" and st in ("tool", "malware") and dt == "attack-pattern":
            g.add((d, ATTACK.hasSoftware, s))
            g.add((s, ATTACK.implementsTechnique, d))
            counts["software-technique"] += 1

    version = next((o.get("x_mitre_version") for o in bundle["objects"] if o["type"] == "x-mitre-collection"), None)
    summary = {"attack_version": version, "triples": len(g), "relationships": counts,
               "techniques": sum(1 for o in objects if o["type"] == "attack-pattern"),
               "mitigations": sum(1 for o in objects if o["type"] == "course-of-action"),
               "groups": sum(1 for o in objects if o["type"] == "intrusion-set"),
               "software": sum(1 for o in objects if o["type"] in ("tool", "malware")),
               "tactics": len(tactic_by_short)}
    return g, summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("bundle", type=Path, help="ATT&CK STIX 2.1 bundle (enterprise-attack-<version>.json)")
    parser.add_argument("output", type=Path, help="Turtle file to write")
    args = parser.parse_args(argv)
    graph, summary = build(args.bundle)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    header = (f"# MITRE ATT&CK Enterprise {summary['attack_version']} in the SEPSES ATT&CK vocabulary.\n"
              f"# Generated by mcp-cskg-rdf/build_attack_rdf.py from {args.bundle.name}; do not edit by hand.\n"
              f"# ATT&CK content (c) The MITRE Corporation, used under the ATT&CK Terms of Use.\n")
    args.output.write_text(header + graph.serialize(format="turtle"), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
