#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
A script to perform logical reasoning on structured data using the Z3 SMT solver.

This script reads a JSON or JSONL file from standard input. It processes all
entries and outputs a SINGLE JSON ARRAY containing all the result objects to
standard output.

All output, including warnings and errors, is formatted as JSON.

The reasoning process follows these steps:
1. Check if the premises (facts and rules) entail the conjecture (P entails C).
   This is equivalent to checking if `Premises AND NOT(Conjecture)` is unsatisfiable.
2. Check if the premises entail the negation of the conjecture (P entails ~C).
   This is equivalent to checking if `Premises AND Conjecture` is unsatisfiable.

The final label is determined by the results of these two checks:
- P entails C AND P entails ~C   => "Self-Contradictory" (The premises are inconsistent)
- P entails C AND NOT(P entails ~C) => "True"
- NOT(P entails C) AND P entails ~C => "False"
- NOT(P entails C) AND NOT(P entails ~C) => "Unknown"

The output includes the determined label and, if applicable, the "unsat core"
which provides a proof for the conclusion.

Requires the 'z3-solver' package. Install it using:
pip install z3-solver

Usage:
  python z3_reasoner.py < input.json > output.json
  cat input.jsonl | python z3_reasoner.py > output.json
"""

import json
import re
import sys
from z3 import Solver, Bool, Not, Or, And, unsat

# Regex to parse atoms, now handling optional `\text{}` wrappers.
ATOM_RE = re.compile(r"(?:\\text{)?([A-Za-z]+)(?:})?\(([^,)]+),\s*(?:\\text{)?(True|False)(?:})?\)")

def parse_atom(atom_str):
    """Parses a string like 'Jompus(Fae, True)' into (Predicate, Term, Value)."""
    clean_atom_str = atom_str.strip()
    match = ATOM_RE.match(clean_atom_str)
    if not match:
        raise ValueError(f"Could not parse atom: {atom_str}")
    
    predicate, term, value_str = match.groups()
    term = term.strip()
    if term == '$x':
        term = 'x'
    return predicate, term, value_str == "True"

def parse_and_or_string(rules_str):
    """
    Robustly parses the 'and_or' string into a list of clean, standardized
    clause bodies (disjunctions). It specifically looks for `\forall x`
    clauses to avoid parsing noisy text or duplicate formats.
    """
    clauses_content = []
    if not rules_str:
        return clauses_content

    for line in rules_str.split('\n'):
        line = line.strip()
        if not line:
            continue

        # We will ONLY process lines that explicitly contain a universal quantifier.
        # This makes the parser robust against descriptive text, implication formats, etc.
        quantifier_pos = line.find('\\forall x')
        if quantifier_pos != -1:
            start_paren_pos = line.find('(', quantifier_pos)
            if start_paren_pos != -1:
                # Find the corresponding closing parenthesis for a well-formed clause
                end_paren_pos = line.rfind(')')
                if end_paren_pos > start_paren_pos:
                    clause_body = line[start_paren_pos + 1 : end_paren_pos].strip()
                    # Final sanity check: ensure it looks like a disjunction
                    if '\\lor' in clause_body and '\\implies' not in clause_body:
                        clauses_content.append(clause_body)

    # Using a set to remove duplicate rules that appear multiple times in the input
    return list(set(clauses_content))

def get_z3_var(predicate, constant, z3_vars):
    """Creates or retrieves a Z3 boolean variable for a ground atom."""
    var_name = f"{predicate}__{constant}"
    if var_name not in z3_vars:
        z3_vars[var_name] = Bool(var_name)
    return z3_vars[var_name]

def atom_to_z3(predicate, constant, value, z3_vars):
    """Converts a parsed ground atom to a Z3 literal."""
    z3_var = get_z3_var(predicate, constant, z3_vars)
    return z3_var if value else Not(z3_var)

def process_json_object(data):
    """Processes a single JSON object with Z3 reasoning."""
    z3_vars = {}
    premises = []
    warnings = []
    item_id = data.get("id", "N/A")

    # 1. Extract constants
    constants = set()
    fact_str = data.get("translated_context", {}).get("Translated_Facts", "")
    if ":::" in fact_str:
        try:
            _, c, _ = parse_atom(fact_str.split(":::")[-1].strip())
            constants.add(c)
        except ValueError: pass
    try:
        _, c, _ = parse_atom(data.get("normalized_conjecture", ""))
        constants.add(c)
    except (ValueError, KeyError): pass

    rules_str = data.get("normalized_context", {}).get("and_or", "")
    clauses_content = parse_and_or_string(rules_str)
    for clause_body in clauses_content:
        for lit_str in [s.strip() for s in clause_body.split('\\lor')]:
            try:
                _, term, _ = parse_atom(lit_str)
                if term != 'x':
                    constants.add(term)
            except ValueError: pass
    if not constants:
        warnings.append("No constants found. Grounding will be empty.")

    # 2. Parse Facts
    if ":::" in fact_str:
        fact_text = fact_str.split(":::")[-1].strip()
        try:
            p, c, v = parse_atom(fact_text)
            premises.append({"name": "fact_0", "z3": atom_to_z3(p, c, v, z3_vars), "text": fact_text})
        except ValueError as e:
            warnings.append(f"Could not parse fact '{fact_text}': {e}")

    # 3. Parse and Ground Rules
    for i, clause_body in enumerate(clauses_content):
        literals_str = [s.strip() for s in clause_body.split('\\lor')]
        is_universal_rule = False
        try:
            parsed_lits = [parse_atom(lit) for lit in literals_str]
            if any(term == 'x' for _, term, _ in parsed_lits):
                is_universal_rule = True
        except ValueError:
            warnings.append(f"Skipping unparsable clause '{clause_body}'")
            continue

        if not is_universal_rule:
            try:
                ground_literals_z3 = [atom_to_z3(p, t, v, z3_vars) for p, t, v in parsed_lits]
                if ground_literals_z3:
                    clause_z3 = Or(ground_literals_z3) if len(ground_literals_z3) > 1 else ground_literals_z3[0]
                    premises.append({"name": f"rule_as_fact_{i}", "z3": clause_z3, "text": clause_body})
            except ValueError:
                warnings.append(f"Skipping unparsable ground clause '{clause_body}'")
            continue

        for const in constants:
            ground_literals_z3, ground_literals_text = [], []
            for p, term, v in parsed_lits:
                current_const = const if term == 'x' else term
                ground_literals_z3.append(atom_to_z3(p, current_const, v, z3_vars))
                ground_literals_text.append(f"{p}({current_const}, {v})")
            clause_z3 = Or(ground_literals_z3) if len(ground_literals_z3) > 1 else ground_literals_z3[0]
            premises.append({"name": f"rule_{i}_ground_{const}", "z3": clause_z3, "text": " or ".join(ground_literals_text)})

    # 4. Parse Conjecture
    try:
        conj_text = data["normalized_conjecture"]
        p, t, v = parse_atom(conj_text)
        conjecture_z3 = atom_to_z3(p, t, v, z3_vars)
        neg_conjecture_z3 = Not(conjecture_z3)
    except (ValueError, KeyError) as e:
        result = {"id": item_id, "solver_label": "Error", "error": f"Could not parse conjecture: {e}"}
        if warnings: result["warnings"] = warnings
        return result

    # 5. Perform Z3 Checks based on the definition of entailment
    
    # Check 1: Do the premises entail the conjecture? (Is P & ~C unsatisfiable?)
    s_entails = Solver()
    s_entails.set("core.minimize", True)
    for p in premises:
        s_entails.assert_and_track(p["z3"], p["name"])
    s_entails.assert_and_track(neg_conjecture_z3, "negated_conjecture")
    
    entails_conjecture = (s_entails.check() == unsat)
    entailment_core = s_entails.unsat_core() if entails_conjecture else []

    # Check 2: Do the premises entail the negation of the conjecture? (Is P & C unsatisfiable?)
    s_neg_entails = Solver()
    s_neg_entails.set("core.minimize", True)
    for p in premises:
        s_neg_entails.assert_and_track(p["z3"], p["name"])
    s_neg_entails.assert_and_track(conjecture_z3, "conjecture")
    
    entails_neg_conjecture = (s_neg_entails.check() == unsat)
    neg_entailment_core = s_neg_entails.unsat_core() if entails_neg_conjecture else []

    # 6. Determine final label based on the two checks
    if entails_conjecture and entails_neg_conjecture:
        # This implies the premises themselves are self-contradictory.
        # We can find the core contradiction from the premises alone.
        s_premise_only = Solver()
        s_premise_only.set("core.minimize", True)
        for p in premises:
            s_premise_only.assert_and_track(p["z3"], p["name"])
        contradiction_core = []
        if s_premise_only.check() == unsat:
            contradiction_core = [str(c) for c in s_premise_only.unsat_core()]
        
        return {
            "id": item_id,
            "solver_label": "Self-Contradictory",
            "core": contradiction_core or [str(c) for c in entailment_core], # Fallback core
            "warnings": warnings or None
        }
    elif entails_conjecture:
        return {
            "id": item_id,
            "solver_label": "True",
            "core": [str(c) for c in entailment_core],
            "warnings": warnings or None
        }
    elif entails_neg_conjecture:
        return {
            "id": item_id,
            "solver_label": "False",
            "core": [str(c) for c in neg_entailment_core],
            "warnings": warnings or None
        }
    else:
        return {
            "id": item_id,
            "solver_label": "Unknown",
            "core": None,
            "warnings": warnings or None
        }


def main():
    """Main: read JSON/JSONL from stdin, process, write JSON array to stdout."""
    all_results = []
    input_str = sys.stdin.read()
    try:
        data = json.loads(input_str)
        items = data if isinstance(data, list) else [data]
        for it in items:
            all_results.append(process_json_object(it))
    except json.JSONDecodeError:
        lines = input_str.strip().split('\n')
        for i, line in enumerate(lines):
            if not line: continue
            try:
                obj = json.loads(line)
                all_results.append(process_json_object(obj))
            except json.JSONDecodeError as e:
                all_results.append({"id": None, "solver_label": "Error", "error": f"Invalid JSON line {i+1}: {e}"})
    print(json.dumps(all_results, indent=2))

if __name__ == "__main__":
    main()