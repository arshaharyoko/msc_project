/**
 * join_prices.js
 *
 * Step 2: joins data/price_template.csv (after you've filled in real prices from
 * vendor sites — copy-pasted by hand, not scraped) back into
 * database_preprocessed.json, adding ONE flat numeric field per component:
 *
 *   cost = (price / pack_size_in_grams) * (concentration_mg / 1000)
 *
 * i.e. `cost` is the $ (currency depends on what you entered in the CSV —
 * keep it consistent across rows, e.g. all USD) contribution of THIS
 * ingredient at THIS formulation's concentration, per liter of medium — not
 * just the raw vendor pack price. Every component in the dataset uses
 * "mg/L", so that's the only conversion implemented.
 *
 * `cost` is set to `null` (not a fabricated number) when it can't be computed:
 *   - concentration === null (ingredient is part of an undisclosed-concentration
 *     supplement premix, e.g. "B27 Supplement" — no individual concentration
 *     exists to multiply by)
 *   - unit isn't "mg/L" (none currently in the dataset, but guarded anyway)
 *   - no price has been entered yet for that component in the CSV (field is
 *     left untouched entirely in this case — see counts printed at the end)
 *
 * Usage:
 *   node data/scripts/join_prices.js
 *
 * Output:
 *   data/database_with_cost.json   (original file is never modified in place)
 */

const fs = require('fs');
const path = require('path');

const DB_PATH = path.join(__dirname, '..', 'database_preprocessed.json');
const CSV_PATH = path.join(__dirname, '..', 'price_template.csv');
const OUT_PATH = path.join(__dirname, '..', 'database_with_cost.json');

const MASS_TO_GRAMS = {
  g: 1,
  kg: 1000,
  mg: 0.001,
  ug: 0.000001,
  'µg': 0.000001,
  mcg: 0.000001,
};

function parseCsv(text) {
  const rows = [];
  let row = [];
  let field = '';
  let inQuotes = false;

  for (let i = 0; i < text.length; i++) {
    const ch = text[i];
    if (inQuotes) {
      if (ch === '"') {
        if (text[i + 1] === '"') {
          field += '"';
          i++;
        } else {
          inQuotes = false;
        }
      } else {
        field += ch;
      }
    } else if (ch === '"') {
      inQuotes = true;
    } else if (ch === ',') {
      row.push(field);
      field = '';
    } else if (ch === '\n' || ch === '\r') {
      if (ch === '\r' && text[i + 1] === '\n') i++;
      row.push(field);
      rows.push(row);
      row = [];
      field = '';
    } else {
      field += ch;
    }
  }
  if (field.length || row.length) {
    row.push(field);
    rows.push(row);
  }
  return rows.filter((r) => r.length > 1 || r[0] !== '');
}

function main() {
  console.log('Reading database:', DB_PATH);
  const db = JSON.parse(fs.readFileSync(DB_PATH, 'utf-8'));

  console.log('Reading price template:', CSV_PATH);
  const rows = parseCsv(fs.readFileSync(CSV_PATH, 'utf-8'));
  const header = rows[0];
  const idx = Object.fromEntries(header.map((h, i) => [h, i]));

  const priceMap = new Map(); // component -> { pricePerGram, currency, vendor, ... }
  let priced = 0;
  let skippedEmpty = 0;
  let skippedBadUnit = 0;

  for (let i = 1; i < rows.length; i++) {
    const r = rows[i];
    const component = r[idx.component];
    const priceStr = r[idx.price];
    const packSizeStr = r[idx.pack_size_value];
    const packUnitStr = (r[idx.pack_size_unit] || '').trim().toLowerCase();

    if (!priceStr?.trim() || !packSizeStr?.trim()) {
      skippedEmpty++;
      continue;
    }

    const price = Number(priceStr);
    const packSize = Number(packSizeStr);
    if (Number.isNaN(price) || Number.isNaN(packSize) || packSize <= 0) {
      console.warn(`Skipping "${component}": price/pack_size not valid numbers.`);
      skippedEmpty++;
      continue;
    }

    const gramsPerUnit = MASS_TO_GRAMS[packUnitStr];
    if (!gramsPerUnit) {
      console.warn(
        `Skipping "${component}": unrecognized pack_size_unit "${packUnitStr}" (expected g/kg/mg/ug).`
      );
      skippedBadUnit++;
      continue;
    }

    const packSizeGrams = packSize * gramsPerUnit;
    const pricePerGram = price / packSizeGrams;

    priceMap.set(component, {
      pricePerGram,
      currency: r[idx.price_currency] || null,
      vendor: r[idx.vendor] || null,
      vendor_product_url: r[idx.vendor_product_url] || null,
      name: r[idx.name] || null,
      cas: r[idx.cas] || null,
      pubchem_cid: r[idx.pubchem_cid] || null,
    });
    priced++;
  }

  console.log(
    `Price references parsed: ${priced} | empty/unfilled: ${skippedEmpty} | bad unit: ${skippedBadUnit}`
  );

  let costed = 0;
  let noConcentration = 0;
  let noPriceRef = 0;
  let unsupportedUnit = 0;

  db.forEach((product) => {
    (product.components || []).forEach((c) => {
      const ref = priceMap.get(c.component);

      if (!ref) {
        c.cost = null; // no price entered for this component yet in the CSV
        noPriceRef++;
        return;
      }

      if (c.concentration === null || c.concentration === undefined) {
        // Part of an undisclosed-concentration supplement premix (e.g. B27) —
        // there's no concentration to multiply by, so no fabricated number.
        c.cost = null;
        noConcentration++;
        return;
      }

      if (c.unit !== 'mg/L') {
        console.warn(
          `Component "${c.component}": unit "${c.unit}" not supported by the mg/L->cost conversion; left cost as null.`
        );
        c.cost = null;
        unsupportedUnit++;
        return;
      }

      const concentrationGrams = c.concentration / 1000; // mg -> g
      c.cost = concentrationGrams * ref.pricePerGram; // $ (ref.currency) per liter of medium, at this instance's concentration
      costed++;
    });
  });

  fs.writeFileSync(OUT_PATH, JSON.stringify(db, null, 2), 'utf-8');
  console.log(`Wrote ${OUT_PATH}`);
  console.log(`Component instances costed:            ${costed}`);
  console.log(`Skipped - no concentration (mix):       ${noConcentration}`);
  console.log(`Skipped - unsupported unit:             ${unsupportedUnit}`);
  console.log(`Skipped - no price entered yet:         ${noPriceRef}`);
}

main();
