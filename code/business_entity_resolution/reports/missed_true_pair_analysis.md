# Blocking Failure Forensics & Root-Cause Classification

## 1. Executive Summary

- **Total True Match Pairs Evaluated:** 86,232
- **Recovered by Union D (Standard):** 73,802 (**85.59%**)
- **Missed True Pairs:** 12,430 (**14.41%**)
- **Misses Caused by Pruning / Candidate Caps:** 23 (**0.03% of all true matches**)

## 2. Root-Cause Failure Classification

| Category | Description | Missed Pairs | % of Missed | % of Ground Truth |
|---|---|---|---|---|
| `D_name_abbreviation_or_acronym` | Severe abbreviation, acronym, or initials without common token | **5,534** | **44.52%** | 6.42% |
| `G_transliteration_script_mismatch` | Devanagari vs Latin script mismatch in India partition | **3,142** | **25.28%** | 3.64% |
| `H_missing_address_on_target` | Target record has empty address field; address blocks cannot fire | **1,019** | **8.2%** | 1.18% |
| `E_token_insertion_deletion` | One name is a proper subset of the other with added/omitted descriptor | **795** | **6.4%** | 0.92% |
| `B_high_addr_sim_but_divergent_name` | Address similarity ≥ 0.80, but business trade name differs significantly | **625** | **5.03%** | 0.72% |
| `C_moderate_name_and_address_similarity` | Both name and address moderately similar (name ~0.60–0.75, address ~0.50) | **461** | **3.71%** | 0.53% |
| `J_missing_house_number_on_target` | S1 has house number, but target has street name only | **440** | **3.54%** | 0.51% |
| `A_high_name_sim_but_divergent_address` | Name similarity ≥ 0.80, but address differs or is partial (no address block) | **156** | **1.26%** | 0.18% |
| `L_very_short_name_collision` | Short name (≤ 5 chars) with high ambiguity | **134** | **1.08%** | 0.16% |
| `I_missing_postal_code_on_target` | S1 has PIN/ZIP code, but target address lacks postal code | **91** | **0.73%** | 0.11% |
| `O_bucket_or_candidate_pruning` | Shared a blocking key, but bucket size exceeded cap (pruned as oversized) | **23** | **0.19%** | 0.03% |
| `F_word_order_transposition` | Identical tokens in different syntactic order | **10** | **0.08%** | 0.01% |

## 3. Pruning Impact Analysis (Step 3)

When `max_bucket_size` was raised from 1,000 to 50,000 and `max_candidates` was increased to 5,000:

- Exactly **23 previously missed pairs were recovered** (+0.03% recall).
- **Keys most responsible for pruned true matches:**

## 4. Concrete Examples of Top Failure Modes

### Mode: `A_high_name_sim_but_divergent_address`

**Example 1 (US):**
- S1: `S1-83688453` | **Titech Holdings LP** | 437 Sires Run Drive, Ostrander, OH
- Match: `S2-308224919` | **Tiech Holdings LP** | SIRES RUN DR, OSTRANDER, OH
- Metrics: Name Sim = `0.944`, Addr Sim = `0.353`, Name Overlap = `0.667`

**Example 2 (India):**
- S1: `S1-897289354` | **YD Solutions Private Limited** | S.No. 171/1, 1St Floor, Hulisiddegowdanadoddi, Girenahalli Post, Harohalli Hobli, 2Nd Ph, Ase, Kiadb, Kanakapura, Bangalore Rural, Karnataka
- Match: `S3-25981289` | **YD Solutions Priavte Limited** | S.no. C-171/1, Kanakapura, Bangalore Rural, ಕರ್ನಾಟಕ
- Metrics: Name Sim = `0.929`, Addr Sim = `0.279`, Name Overlap = `1.0`

**Example 3 (India):**
- S1: `S1-182135610` | **Golden Agro Private Limited** | Telangana, Begumpet, 7-1-24, Hyderabad, Block B, Ground Floor, Roxana Towers Greenlands
- Match: `S3-172636518` | **M/s Golden Arno Private Limited** | H.no 780 7-1-24, Block B, Ground Floor, Roxana Towers Greenlands, Begumpet, Hyderabad, Andhra Pradesh
- Metrics: Name Sim = `0.806`, Addr Sim = `0.347`, Name Overlap = `0.5`

### Mode: `H_missing_address_on_target`

**Example 1 (India):**
- S1: `S1-497864956` | **Next Properties Private Limited** | 703-A, 7Th Floor, Skyline Epitome, Kirol Road Vidhyavihar, Nr. Jolly Gymkhana, Ghatkop, Ar (West), Mumbai, Mumbai City, Maharashtra
- Match: `S3-68275427` | **Next Properties Properties Private Limited** | 
- Metrics: Name Sim = `0.738`, Addr Sim = `0.0`, Name Overlap = `1.0`

**Example 2 (India):**
- S1: `S1-998633222` | **Fortune Management Private Limited** | Flat No 2, Meera Bldg, Seema Chs Ltd, Ghatla Village Road, Chembur, Mumbai, Maharashtra
- Match: `S3-564769879` | **Sri Fortune Private Management Limited** | 
- Metrics: Name Sim = `0.474`, Addr Sim = `0.0`, Name Overlap = `1.0`

**Example 3 (US):**
- S1: `S1-366774065` | **Zinn Golden Photonics Center** | 913 3530 Road E, Neoga, IL
- Match: `S3-340125107` | **zinn golden photonics center - 7418502269** | 
- Metrics: Name Sim = `0.585`, Addr Sim = `0.0`, Name Overlap = `1.0`

### Mode: `C_moderate_name_and_address_similarity`

**Example 1 (US):**
- S1: `S1-231212454` | **Romy Bresee Storage** | 5955 W A Street, West Linn, OR
- Match: `S3-606879311` | **Romybreseestorage.Com** | Oregon, 5955 West A Street, West Linn
- Metrics: Name Sim = `0.619`, Addr Sim = `0.595`, Name Overlap = `0.0`

**Example 2 (India):**
- S1: `S1-72981612` | **Rr Industries Group Limited** | Eastern High, T9/F10/E1 Ag-1, New Town, Kolkata, Kolkata, Howrah, West Bengal
- Match: `S2-583199424` | **M/s Rr Indusnreeis Group Limited** | EASTERN HIGH, T9/F10/E1 AG-1, NEW TOWN, KOLKATA, KOLKATA, West Bengal
- Metrics: Name Sim = `0.781`, Addr Sim = `0.545`, Name Overlap = `0.5`

**Example 3 (US):**
- S1: `S1-700039521` | **Fenet LLC** | 876 Cumbleside Lane, Unit K, Westerville, OH
- Match: `S3-196776668` | **Fetn LLC** | 876 Cumbleside Ln, Unit K, Westerville, Ohio
- Metrics: Name Sim = `0.667`, Addr Sim = `0.886`, Name Overlap = `0.0`

### Mode: `D_name_abbreviation_or_acronym`

**Example 1 (India):**
- S1: `S1-322278645` | **Tech Services Private Limited** | 601/C Sahajanand Complexnr New Madhavpura Market Shahibaug, Ahmedabad, Gujarat
- Match: `S2-843869139` | **ટેક સર્વિસીસ પ્રાઇવેટ લિમિટેડ** | H.NO #601/C SAHAJANAND COMPLEXNR NEW MADHAVPURA MARKET SHAHIBAUG, AHMEDABAD, Gujarat
- Metrics: Name Sim = `0.069`, Addr Sim = `0.345`, Name Overlap = `0.0`

**Example 2 (India):**
- S1: `S1-197025412` | **Anilika Energy Private Limited** | Creware Coworks, No. 4, 2Nd And 3Rd Floor, Himagiri Silicon City, Thogur Cross, Phase 1, Bangalore South, Bangalore, Karnataka
- Match: `S2-666892279` | **anilikaenergy.com** | CREWARE COWORKS, NO. 4, 2ND AND 3RD FLOOR, HIMAGIRI SILICON CITY, THOGUR CROSS, PHASE 1, BANGALORE SOUTH, Karnataka
- Metrics: Name Sim = `0.4`, Addr Sim = `0.421`, Name Overlap = `0.0`

**Example 3 (India):**
- S1: `S1-244643323` | **Good Life Services Pvt Ltd** | Plot No. 100-101, Sector-35, Hsiidc Udyog Vihar, Phase-Vii, Sadar Bazar, Gurgaon, Haryana
- Match: `S3-421311396` | **goodlifeservices.com** | Plot No. 100-101, Sector-35, Hsiidc Udyog Vihar, Phase-vii, Sadar Bazar, Gurgaon, HR
- Metrics: Name Sim = `0.5`, Addr Sim = `0.921`, Name Overlap = `0.0`

