/**
 * regen_csv.js
 *
 * Rebuilds data/price_template.csv from the already-populated pubchem_cache.json
 * (no network calls) with cleaner NUMERIC price columns so join_prices.js can
 * actually compute cost = price_per_gram * concentration, instead of just
 * dumping the vendor's raw pack price.
 *
 * You fill in: price, price_currency, pack_size_value, pack_size_unit (g/kg/mg/ug),
 * vendor, vendor_product_url.
 *
 * Usage: node data/scripts/regen_csv.js
 */

const fs = require('fs');
const path = require('path');

const DB_PATH = path.join(__dirname, '..', 'database_preprocessed.json');
const CACHE_PATH = path.join(__dirname, '..', 'pubchem_cache.json');
const OUT_CSV = path.join(__dirname, '..', 'price_template.csv');

function csvEscape(value) {
  const s = String(value ?? '');
  if (/[",\n]/.test(s)) return `"${s.replace(/"/g, '""')}"`;
  return s;
}

function buildVendorSearchUrl(kind, name, cas) {
  if (kind === 'uniprot') {
    return `https://www.rndsystems.com/search?keywords=${encodeURIComponent(name || '')}`;
  }
  return `https://www.sigmaaldrich.com/US/en/search/${encodeURIComponent(cas || name || '')}`;
}

function main() {
  const db = JSON.parse(fs.readFileSync(DB_PATH, 'utf-8'));
  const cache = JSON.parse(fs.readFileSync(CACHE_PATH, 'utf-8'));

  // frequency + whether this component ever appears WITH a concentration
  // (some components appear both as individually-dosed AND as part of an
  // undisclosed-concentration supplement mix like B27 — both are worth pricing,
  // the per-instance cost calc in join_prices.js handles which applies).
  const freq = new Map();
  const anyConcentration = new Map();
  db.forEach((p) => {
    (p.components || []).forEach((c) => {
      freq.set(c.component, (freq.get(c.component) || 0) + 1);
      anyConcentration.set(
        c.component,
        (anyConcentration.get(c.component) || false) || c.concentration !== null
      );
    });
  });

  const header = [
    'component',
    'component_source',
    'frequency_in_dataset',
    'has_dosed_instances', // true if at least one formulation gives it a concentration
    'pubchem_cid',
    'name',
    'cas',
    'molecular_formula',
    'molecular_weight_g_per_mol',
    'price', // numeric, total price for the pack
    'price_currency', // e.g. USD
    'pack_size_value', // numeric, e.g. 100
    'pack_size_unit', // g | kg | mg | ug
    'vendor',
    'vendor_product_url',
    'vendor_search_hint',
    'notes',
  ];

  const lines = [header.join(',')];
  for (const [component, count] of freq.entries()) {
    const r = cache[component] || {};
    const row = [
      component,
      r.kind || '',
      count,
      anyConcentration.get(component) ? 'yes' : 'no',
      r.cid || '',
      r.name || '',
      r.cas || '',
      r.formula || '',
      r.molecular_weight || '',
      '',
      '',
      '',
      '',
      '',
      '',
      buildVendorSearchUrl(r.kind, r.name, r.cas),
      r.error || '',
    ];
    lines.push(row.map(csvEscape).join(','));
  }

  fs.writeFileSync(OUT_CSV, lines.join('\n'), 'utf-8');
  console.log(`Wrote ${OUT_CSV} (${lines.length - 1} components)`);
}

main();
