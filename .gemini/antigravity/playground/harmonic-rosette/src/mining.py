import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from mlxtend.frequent_patterns import fpgrowth, association_rules
from src.cleaner import load_and_clean_data
import pickle
import os

CACHE_PATH = "data/rules_cache.pkl"

def mine_rules(min_support=0.01, min_confidence=0.1):
    """
    Mines association rules from the recipe dataset.
    """
    print("Loading and cleaning data...")
    transactions, _ = load_and_clean_data()
    
    if not transactions:
        print("No transactions found.")
        return None

    print(f"Encoding {len(transactions)} transactions...")
    te = TransactionEncoder()
    te_ary = te.fit(transactions).transform(transactions)
    df = pd.DataFrame(te_ary, columns=te.columns_)

    print(f"Running FP-Growth with min_support={min_support}...")
    frequent_itemsets = fpgrowth(df, min_support=min_support, use_colnames=True)
    
    print(f"Found {len(frequent_itemsets)} frequent itemsets.")
    if frequent_itemsets.empty:
        return pd.DataFrame()

    print(f"Generating rules with min_confidence={min_confidence}...")
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    
    # Add 'lift' and sort
    rules = rules.sort_values(['lift', 'confidence'], ascending=[False, False])
    
    print(f"Generated {len(rules)} rules.")
    return rules

def get_recommendations(rules, ingredients, top_k=5):
    """
    Given a set of ingredients, recommend associated items.
    """
    # Simple logic: Find rules where antecedent is a subset of ingredients
    # This matches: if {A, B} -> {C}, and user has {A, B}, suggest C.
    
    # Filter rules where antecedents overlap with input
    # Note: rigorous subset matching might be slow if we have many rules.
    # We can do exact match or partial match.
    
    # converting frozensets to sets for easier handling
    if rules is None or rules.empty:
        return []

    input_set = set([i.lower() for i in ingredients])
    
    suggestions = []
    
    # Heuristic: Check rules where consequent is NOT in input
    # and antecedent is in input (or close to it)
    
    # Relaxed Logic: Check rules where antecedent has ANY overlap with input
    # This allows users to find associations even if they don't have the full antecedent
    relevant_rules = rules[rules['antecedents'].apply(lambda x: not x.isdisjoint(input_set))]
    
    for _, row in relevant_rules.iterrows():
        consequents = row['consequents']
        for item in consequents:
            if item not in input_set:
                suggestions.append({
                    'item': item,
                    'confidence': row['confidence'],
                    'lift': row['lift'],
                    'rule': f"{set(row['antecedents'])} -> {set(row['consequents'])}"
                })
    
    # Deduplicate and sort by lift
    seen = set()
    unique_suggestions = []
    for s in suggestions:
        if s['item'] not in seen:
            seen.add(s['item'])
            unique_suggestions.append(s)
    
    unique_suggestions.sort(key=lambda x: x['lift'], reverse=True)
    return unique_suggestions[:top_k]

if __name__ == "__main__":
    rules = mine_rules(min_support=0.005) # Lower support for more interesting result
    if rules is not None and not rules.empty:
        print(rules.head())
        # Test recommendation
        test_ingredients = ["chicken", "garlic"]
        recs = get_recommendations(rules, test_ingredients)
        print(f"\nRecommendations for {test_ingredients}:")
        for r in recs:
            print(f"- {r['item']} (Lift: {r['lift']:.2f})")
