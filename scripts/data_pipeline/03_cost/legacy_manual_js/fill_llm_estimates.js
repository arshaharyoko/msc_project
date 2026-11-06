/**
 * fill_llm_estimates.js
 *
 * Fills data/price_template.csv with price estimates from general chemistry/
 * biochem-vendor-catalog knowledge (typical Sigma-Aldrich / R&D Systems academic
 * bulk pricing), so the pipeline is runnable immediately without manual entry.
 *
 * IMPORTANT: these are NOT scraped/live vendor prices. They are ballpark figures
 * from general knowledge, all in USD, meant to unblock a first runnable pass.
 * Every filled row is tagged in `notes` as an estimate. Verify against a real
 * vendor quote before treating these numbers as ground truth for anything
 * decision-critical (cost comparisons for a paper, purchasing, etc.).
 *
 * Components I'm not confident identifying correctly (obscure PDB-ligand-style
 * names, ambiguous complex peptides) are left blank rather than guessed.
 *
 * Usage: node data/scripts/fill_llm_estimates.js
 */

const fs = require('fs');
const path = require('path');

const CSV_PATH = path.join(__dirname, '..', 'price_template.csv');

// key: CAS number (preferred) or exact `component` SMILES/accession when CAS is missing
// value: [price_usd, pack_size_value, pack_size_unit]
const ESTIMATES = {
  // --- Amino acids ---
  '56-40-6': [40, 500, 'g'], // Glycine
  '56-41-7': [45, 100, 'g'], // L-Alanine
  '74-79-3': [60, 100, 'g'], // L-Arginine
  '70-47-3': [70, 100, 'g'], // L-Asparagine
  '56-84-8': [50, 100, 'g'], // L-Aspartic acid
  '52-90-4': [60, 25, 'g'], // L-Cysteine
  '56-89-3': [90, 25, 'g'], // L-Cystine
  '56-86-0': [45, 100, 'g'], // L-Glutamic acid
  '56-85-9': [60, 100, 'g'], // L-Glutamine
  '71-00-1': [70, 25, 'g'], // L-Histidine
  '73-32-5': [75, 25, 'g'], // L-Isoleucine
  '61-90-5': [60, 25, 'g'], // L-Leucine
  '56-87-1': [65, 25, 'g'], // L-Lysine
  '63-68-3': [55, 25, 'g'], // L-Methionine
  '63-91-2': [55, 25, 'g'], // L-Phenylalanine
  '147-85-3': [60, 25, 'g'], // L-Proline
  '56-45-1': [50, 25, 'g'], // L-Serine
  '72-19-5': [70, 25, 'g'], // L-Threonine
  '73-22-3': [90, 25, 'g'], // L-Tryptophan
  '60-18-4': [60, 25, 'g'], // L-Tyrosine
  '72-18-4': [60, 25, 'g'], // L-Valine
  '51-35-4': [60, 25, 'g'], // Hydroxyproline
  '39537-23-0': [90, 25, 'g'], // L-Alanyl-L-glutamine
  '30925-07-6': [70, 25, 'g'], // Cystine dihydrochloride
  '52-89-1': [70, 25, 'g'], // L-Cysteine HCl

  // --- Vitamins / cofactors ---
  '58-85-5': [60, 1, 'g'], // Biotin
  '67-48-1': [40, 100, 'g'], // Choline chloride
  '79-83-4': [70, 25, 'g'], // D-Pantothenic acid
  '137-08-6': [70, 25, 'g'], // Calcium D-pantothenate
  '59-30-3': [60, 25, 'g'], // Folic acid
  '87-89-8': [40, 100, 'g'], // myo-Inositol
  '98-92-0': [30, 100, 'g'], // Nicotinamide
  '58-56-0': [40, 25, 'g'], // Pyridoxine HCl
  '65-22-5': [45, 5, 'g'], // Pyridoxal HCl
  '83-88-5': [80, 5, 'g'], // Riboflavin
  '67-03-8': [50, 25, 'g'], // Thiamine HCl
  '68-19-9': [100, 1, 'g'], // Cyanocobalamin (B12)
  '50-81-7': [30, 100, 'g'], // Ascorbic acid (Vit C)
  '23313-12-4': [90, 1, 'g'], // Ascorbic acid 2-phosphate
  '84309-23-9': [90, 1, 'g'], // Ascorbic acid 2-phosphate Mg salt
  '50-14-6': [60, 1, 'g'], // Ergocalciferol (Vit D2)
  '58-27-5': [40, 25, 'g'], // Menadione (Vit K3)
  '10191-41-0': [60, 5, 'g'], // Vitamin E
  '52225-20-4': [50, 5, 'g'], // Vitamin E acetate

  // --- Salts / inorganics ---
  '10043-52-4': [40, 500, 'g'], // CaCl2
  '7758-98-7': [40, 100, 'g'], // CuSO4
  '10421-48-4': [40, 100, 'g'], // Ferric nitrate
  '10028-22-5': [50, 100, 'g'], // Ferric sulfate
  '7786-30-3': [40, 500, 'g'], // MgCl2
  '7487-88-9': [30, 500, 'g'], // MgSO4
  '7447-40-7': [30, 500, 'g'], // KCl
  '144-55-8': [30, 500, 'g'], // NaHCO3
  '7647-14-5': [25, 500, 'g'], // NaCl
  '7558-79-4': [30, 500, 'g'], // Na2HPO4
  '7558-80-7': [30, 500, 'g'], // NaH2PO4
  '7733-02-0': [30, 100, 'g'], // ZnSO4
  '10025-73-7': [50, 25, 'g'], // CrCl3
  '7786-81-4': [45, 100, 'g'], // NiSO4
  '7772-99-8': [50, 100, 'g'], // SnCl2
  '3522-50-7': [40, 100, 'g'], // Ferric ammonium citrate
  '7803-55-6': [60, 25, 'g'], // Ammonium metavanadate
  '7785-87-7': [35, 100, 'g'], // MnSO4
  '12501-45-0': [60, 100, 'g'], // Ammonium molybdate
  '7757-79-1': [30, 100, 'g'], // KNO3
  '10102-18-8': [55, 10, 'g'], // Sodium selenite
  '1344-09-8': [40, 100, 'g'], // Sodium metasilicate
  '7718-54-9': [45, 100, 'g'], // NiCl2
  '7646-85-7': [35, 100, 'g'], // ZnCl2
  '7720-78-7': [35, 100, 'g'], // FeSO4
  '10124-37-5': [35, 100, 'g'], // Ca(NO3)2
  '7783-00-8': [60, 10, 'g'], // Selenious acid
  '7783-08-6': [70, 10, 'g'], // Selenic acid
  '14124-67-5': [70, 5, 'g'], // Selenite (other form)
  sodium_phosphate_no_cas: [30, 500, 'g'], // handled by SMILES key below

  // --- Sugars ---
  '50-99-7': [30, 500, 'g'], // D-Glucose
  '59-23-4': [45, 100, 'g'], // D-Galactose
  '69-65-8': [35, 100, 'g'], // D-Mannitol
  '50-69-1': [60, 25, 'g'], // D-Ribose

  // --- Fatty acids / lipids ---
  '60-33-3': [55, 5, 'g'], // Linoleic acid
  '463-40-1': [55, 5, 'g'], // Linolenic acid
  '112-80-1': [40, 25, 'g'], // Oleic acid
  '57-10-3': [40, 25, 'g'], // Palmitic acid
  '373-49-9': [70, 0.1, 'g'], // Palmitoleic acid (100 mg)
  '57-11-4': [40, 25, 'g'], // Stearic acid
  '544-63-8': [40, 25, 'g'], // Myristic acid
  '506-32-1': [90, 0.005, 'g'], // Arachidonic acid (5 mg)
  '57-88-5': [40, 5, 'g'], // Cholesterol
  '1077-28-7': [55, 5, 'g'], // DL-alpha-Lipoic acid

  // --- Misc small molecules / hormones / drugs ---
  '68-94-0': [60, 25, 'g'], // Hypoxanthine
  '143-74-8': [40, 25, 'g'], // Phenol red
  '110-60-1': [50, 25, 'g'], // Putrescine
  '333-93-7': [55, 25, 'g'], // Putrescine dihydrochloride
  '113-24-6': [40, 100, 'g'], // Sodium pyruvate
  '50-89-5': [55, 5, 'g'], // Thymidine
  '58-27-5_dup': null,
  '50-22-6': [60, 5, 'g'], // Progesterone
  '57-83-0': [60, 1, 'g'], // Medroxyprogesterone acetate
  '6893-02-3': [80, 0.1, 'g'], // T3 (triiodothyronine), 100 mg
  '9004-10-8': [90, 0.1, 'g'], // Insulin, bovine (100 mg)
  '70-18-8': [60, 5, 'g'], // Glutathione
  '541-15-1': [50, 5, 'g'], // L-Carnitine
  '127-47-9': [50, 5, 'g'], // Retinyl acetate (Vit A acetate)
  '85721-33-1': [50, 1, 'g'], // Ciprofloxacin
  '51-43-4': [60, 1, 'g'], // Norepinephrine
  '7365-45-9': [50, 100, 'g'], // HEPES
  '7789-24-4': [50, 25, 'g'], // LiF
  '24305-27-9': [90, 0.005, 'g'], // TRH (5 mg)
  '61-33-6': [40, 5, 'g'], // Penicillin G
  '60-24-2': [30, 111, 'g'], // 2-Mercaptoethanol (~100 mL @ 1.11 g/mL)
  '1403-66-3': [40, 1, 'g'], // Gentamicin
  '66575-29-9': [80, 0.025, 'g'], // Forskolin (25 mg)
  '146986-50-7': [90, 0.005, 'g'], // Y-27632 (5 mg)
  '154447-36-6': [80, 0.025, 'g'], // LY294002 (25 mg)
  '253431-35-5': [90, 0.025, 'g'], // Rapamycin (25 mg)
  '133294-37-8': [40, 1, 'g'], // Resveratrol
  '480-19-3': [40, 25, 'g'], // Quercetin
  '78824-30-3': [90, 0.01, 'g'], // Garcinol (10 mg)
  '106392-12-5': [60, 100, 'g'], // Pluronic-type block copolymer
  heparin_sodium_no_cas: [45, 1, 'g'],
  '67-68-5': [42, 1100, 'g'], // DMSO (~1 L @ 1.1 g/mL)
  '141-43-5': [30, 102, 'g'], // Ethanolamine (~100 mL @ 1.02 g/mL)
  '50-23-7': [40, 5, 'g'], // Hydrocortisone
  '150-13-0': [30, 25, 'g'], // 4-Aminobenzoic acid (PABA)
  '66-22-8': [40, 25, 'g'], // Uracil
  '69-89-6': [60, 5, 'g'], // Xanthine
  '302-79-4': [70, 1, 'g'], // All-trans retinoic acid (ATRA)
  '50-67-9': [60, 1, 'g'], // Serotonin
  '112-63-0': [50, 5, 'g'], // Methyl linoleate
  '7683-59-2': [50, 1, 'g'], // Isoproterenol
  '127-09-3': [30, 500, 'g'], // Sodium acetate
  '1119-34-2': [60, 100, 'g'], // L-Arginine HCl
  '657-27-2': [60, 100, 'g'], // L-Lysine HCl
  '62-49-7': [40, 100, 'g'], // Choline (free base)
  '1397-89-3': [90, 0.1, 'g'], // Amphotericin B (100 mg)
  '59-43-8': [45, 25, 'g'], // Thiamine (free base)
  '535-75-1': [50, 5, 'g'], // Pipecolic acid

  // --- Recombinant proteins / growth factors (uniprot; priced by µg vials — LOW confidence) ---
  P01308: [90, 0.1, 'g'], // Insulin (human recombinant), 100 mg
  P02787: [150, 0.1, 'g'], // Transferrin, 100 mg
  Q8TAT2: [300, 0.00001, 'g'], // FGF-binding protein 3, 10 ug
  P10600: [200, 0.00001, 'g'], // TGF-beta 3, 10 ug
  Q02297: [250, 0.00001, 'g'], // Neuregulin-1, 10 ug
  P01137: [150, 0.00001, 'g'], // TGF-beta 1, 10 ug
  P05019: [200, 0.00005, 'g'], // IGF-1, 50 ug
  P02768: [60, 25, 'g'], // Serum albumin (bulk, cheap per gram)
  P04085: [250, 0.00001, 'g'], // PDGF-A, 10 ug
  P04004: [200, 0.0001, 'g'], // Vitronectin, 100 ug
  P17936: [250, 0.00001, 'g'], // IGFBP3, 10 ug
  P01555: [300, 0.001, 'g'], // Cholera toxin subunit A, 1 mg
  P01133: [150, 0.0002, 'g'], // EGF, 200 ug
  Q9HBE4: [250, 0.00001, 'g'], // IL-21, 10 ug
  P56704: [400, 0.00001, 'g'], // Wnt-3a, 10 ug
  P05231: [150, 0.00001, 'g'], // IL-6, 10 ug
  P09603: [200, 0.00001, 'g'], // CSF1/M-CSF, 10 ug
  P04141: [150, 0.00001, 'g'], // GM-CSF, 10 ug
  O94907: [250, 0.00001, 'g'], // DKK1, 10 ug
  P01375: [150, 0.00001, 'g'], // TNF, 10 ug
  P01579: [150, 0.00001, 'g'], // IFN-gamma, 10 ug
};

// Rows with no CAS in the CSV but that I can confidently identify by exact
// component (SMILES/accession) string instead:
const BY_COMPONENT = {
  '[O-]P(=O)([O-])[O-].[Na+]': ESTIMATES.sodium_phosphate_no_cas, // sodium phosphate
  'CC(=O)NC1C(C(C(OC1O)COS(=O)(=O)O)OC2C(C(C(C(O2)C(=O)O)OC3C(C(C(C(O3)CO)OC4C(C(C(C(O4)C(=O)O)O)O)OS(=O)(=O)O)OS(=O)(=O)O)NS(=O)(=O)O)O)OS(=O)(=O)O)O.[Na+]':
    ESTIMATES.heparin_sodium_no_cas, // heparin sodium
};

function csvParse(text) {
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
        } else inQuotes = false;
      } else field += ch;
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
    } else field += ch;
  }
  if (field.length || row.length) {
    row.push(field);
    rows.push(row);
  }
  return rows.filter((r) => r.length > 1 || r[0] !== '');
}

function csvEscape(v) {
  const s = String(v ?? '');
  if (/[",\n]/.test(s)) return `"${s.replace(/"/g, '""')}"`;
  return s;
}

function main() {
  const text = fs.readFileSync(CSV_PATH, 'utf-8');
  const rows = csvParse(text);
  const header = rows[0];
  const idx = Object.fromEntries(header.map((h, i) => [h, i]));

  let filled = 0;
  for (let i = 1; i < rows.length; i++) {
    const r = rows[i];
    const component = r[idx.component];
    const cas = r[idx.cas];

    let est = null;
    if (cas && ESTIMATES[cas]) est = ESTIMATES[cas];
    else if (ESTIMATES[component]) est = ESTIMATES[component]; // uniprot accessions
    else if (BY_COMPONENT[component]) est = BY_COMPONENT[component];

    if (!est) continue;

    const [price, packVal, packUnit] = est;
    r[idx.price] = String(price);
    r[idx.price_currency] = 'USD';
    r[idx.pack_size_value] = String(packVal);
    r[idx.pack_size_unit] = packUnit;
    r[idx.vendor] = r[idx.component_source] === 'uniprot' ? 'R&D Systems (estimate)' : 'Sigma-Aldrich (estimate)';
    r[idx.notes] =
      'ESTIMATE (LLM general knowledge, NOT scraped/live vendor data) - verify before relying on it for real decisions';
    filled++;
  }

  const out = rows.map((r) => r.map(csvEscape).join(',')).join('\n');
  fs.writeFileSync(CSV_PATH, out, 'utf-8');
  console.log(`Filled ${filled}/${rows.length - 1} components with estimated prices.`);
  console.log(`Left blank (low confidence / couldn't identify): ${rows.length - 1 - filled}`);
}

main();
