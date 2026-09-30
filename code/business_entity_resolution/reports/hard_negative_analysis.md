# False-Merge Hard Negative Discovery

## 1. Discovered Negative Failure Modes

Hard negative candidates generated via token-overlap blocking that are **NOT** true matches in ground truth:

1. **Same/Similar Name, Different Address:** Found 100 high-risk pairs (franchises, chains, common names).
2. **Same Address, Different Business:** Found 10 high-risk pairs (multi-tenant complexes, malls, tech parks).
3. **Short Name Collisions:** Found 67 pairs (2-5 character collisions with high string similarity).

## 2. Concrete Discovered Hard Negative Examples

### Category: `same_name_diff_address`

**Example 1 (US):**
- S1: `S1-766693014` | Name: **Helios Inc.** | Addr: 515 Franru Lane, Unit APT 208, Saint Louis, MO
- Match: `S2-74226485` | Name: **Helios Inc** | Addr: #695 PARK LN, PO BOX 3170, MONROE, WA
- Scores: Name Sim = `0.909`, Addr Sim = `0.217`

**Example 2 (US):**
- S1: `S1-766202209` | Name: **Urgent Care Physicians LLC** | Addr: 21 Mount Pleasant Street, Unit A, Marlborough, MA
- Match: `S2-756065363` | Name: **Urgent Cáre Physicians, LLC** | Addr: Block 800 Canby St, LARAMIE, WY
- Scores: Name Sim = `0.926`, Addr Sim = `0.245`

**Example 3 (US):**
- S1: `S1-141554589` | Name: **Womens Health Care LLC** | Addr: 1465 Navajo Road, Bldg BUILDING, Coalville, UT
- Match: `S2-458860562` | Name: **Womens Health Care LLC** | Addr: KS, TOPEKA, 1201 FILLMORE SAINT
- Scores: Name Sim = `1.0`, Addr Sim = `0.174`

### Category: `same_address_diff_business`

**Example 1 (India):**
- S1: `S1-747065084` | Name: **Infra Surender Birds Private Limited** | Addr: E-8/C-1 Sector-C1, Lda Coly. Kp. Road, Lucknow, Uttar Pradesh
- Match: `S3-119662318` | Name: **Vasani** | Addr: E-8/C-1 Sector-C1, Lda Coly. Kp. Road, Lucknow, Uttar Pradesh
- Scores: Name Sim = `0.111`, Addr Sim = `1.0`

**Example 2 (US):**
- S1: `S1-225036015` | Name: **Person, Christensen and Harnett Flagstar Inc** | Addr: 160 Emerson Place, Camden, AR
- Match: `S3-109077694` | Name: **Pyrairi** | Addr: 160 Emerson Place, Camden, AR
- Scores: Name Sim = `0.091`, Addr Sim = `1.0`

**Example 3 (US):**
- S1: `S1-48571327` | Name: **Frontier Alliance Inc** | Addr: 3400 June Drive, Mckinney, TX
- Match: `S3-999173544` | Name: **Mirawex** | Addr: 3400 June Drive, Mckinney, TX
- Scores: Name Sim = `0.19`, Addr Sim = `1.0`

### Category: `short_name_collisions`

**Example 1 (US):**
- S1: `S1-7381669` | Name: **Aurora** | Addr: 1910 Bridges Street, Morehead City, NC
- Match: `S2-56831669` | Name: **Aurora** | Addr: 92 CURTIS AVE, QUINCY, MA
- Scores: Name Sim = `1.0`, Addr Sim = `0.211`

**Example 2 (US):**
- S1: `S1-233585334` | Name: **Aurora** | Addr: 600 Jacks Mountain Road, Franklin County, VA
- Match: `S2-56831669` | Name: **Aurora** | Addr: 92 CURTIS AVE, QUINCY, MA
- Scores: Name Sim = `1.0`, Addr Sim = `0.182`

**Example 3 (US):**
- S1: `S1-298935178` | Name: **Timber** | Addr: 4945 Mcminnville Highway, TN, Doyle
- Match: `S2-143604259` | Name: **Timber** | Addr: ##7840 OAKLEAF AVE, ELMWOOD PARK, IL
- Scores: Name Sim = `1.0`, Addr Sim = `0.056`

