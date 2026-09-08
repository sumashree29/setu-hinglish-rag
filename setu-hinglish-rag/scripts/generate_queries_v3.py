import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
chunks_file = ROOT / "data" / "processed" / "corpus_chunks_v2.jsonl"
chunks = [json.loads(l) for l in open(chunks_file, encoding='utf-8') if l.strip()]
chunks_map = {c["chunk_id"]: c for c in chunks}
c_chunks = [c for c in chunks if c['chunk_id'].startswith('C')]

def extract_q(text):
    m = re.match(r'Q:\s*(.*?)\nA:', text, re.DOTALL)
    if not m: return None
    q = m.group(1).strip()
    parts = q.split('?')
    return parts[0].strip() + '?' if parts else q

def strip_numbering(q):
    return re.sub(r'^(Q\.?\s*)?[\divxlIVXL]+[\.\)]\s*', '', q, flags=re.IGNORECASE).strip()

def is_structural_header(q):
    junk_patterns = [
        r'^(An?\s+)?illustration:?$', r'^table\b', r'^figure\b', r'^amendments?\b',
        r'^annex', r'^appendix', r'^part\s+[ivxlc]+', r'^chapter\b',
    ]
    for p in junk_patterns:
        if re.match(p, q, re.IGNORECASE):
            return True
    words = q.split()
    has_qword = bool(re.match(r'^(what|who|how|can|is|are|whether|when|should|does|do|will|which)\b', q, re.IGNORECASE))
    if not has_qword and len(words) <= 5:
        return True
    return False

# Domain synonym & phrase simplifications to reduce verbatim copy-paste
PHRASE_SIMPLIFICATIONS = [
    (r"foreign exchange earnings", "foreign exchange"),
    (r"permissible credits into this account", "permissible credits"),
    (r"permissible debits into this account", "permissible debits"),
    (r"different types of eefc accounts", "EEFC account types"),
    (r"eefc account and what are its benefits", "EEFC account features"),
    (r"open an eefc account", "EEFC account opening"),
    (r"special economic zone \(sez\) units", "SEZ units"),
    (r"remittance of assets", "asset remittance"),
    (r"tax implications in respect of remittance of assets", "asset remittance tax"),
    (r"participate in the scheme", "scheme participation"),
    (r"eligible investors participate in the auctions", "auction participation"),
    (r"aggregator or facilitator", "aggregator facilitator"),
    (r"minimum/\s*maximum bidding amount", "bidding limits"),
    (r"bids can an investor make under this scheme", "investor bids allowed"),
    (r"investor make payment for the security", "security payment"),
    (r"non-competitive bidder know the modalities of payment", "non-competitive bidder payment mode"),
    (r"rbi allot the bids to non-competitive bidder", "non-competitive allotment by RBI"),
    (r"securities be issued", "securities issuance"),
    (r"aggregator or facilitator make partial allotment", "partial allotment rules"),
    (r"payment for the securities is made to the aggregator or facilitator after the date of issue of the security", "delayed payment to aggregator"),
    (r"major accounts that can be opened in india by a non-resident", "NRI bank accounts"),
    (r"accounts that a tourist visiting india can open", "tourist bank accounts"),
    (r"deposits that foreign diplomatic missions/ personnel and their family members in india can hold", "diplomatic mission deposits"),
    (r"multilateral organisation have deposits in india", "multilateral organisation deposits"),
    (r"indian company accept deposits from non-residents in compliance with section 160 of the companies act, 2013", "company deposits from non-residents"),
    (r"foreign portfolio investor or a foreign venture capital investor open a foreign currency account in india", "FPI foreign currency account"),
    (r"open an escrow account in india and for what purpose", "Escrow account opening"),
    (r"authorised persons under foreign exchange management act, 1999 \(“fema”\)", "FEMA Authorised Persons"),
    (r"objective behind allowing ffmcs to do business", "FFMC business objective"),
    (r"licence is mandatory for carrying out money changing business", "money changing licence mandate"),
    (r"apply for an ffmc licence", "FFMC licence application"),
    (r"nof required to be maintained on an ongoing basis", "ongoing NOF requirement"),
    (r"time frame within which a newly licensed ffmc should commence operations", "FFMC operations time limit"),
    (r"application for licence renewal be made", "licence renewal application"),
    (r"money changing facilities presently available in india", "money changing facilities"),
    (r"objective of the scheme for appointing franchisees by ad category banks, ads category-ii and ffmcs for undertaking restricted money changing activities", "franchisee appointment objective"),
    (r"salient features of franchisee agreement", "franchisee agreement features"),
    (r"submission of application by ad category – i bank/ ad category - ii/ ffmc to the reserve bank for appointment of franchisees", "franchisee application submission"),
    (r"checks to be ensured by ad category-i banks/ads category-ii/ffmcs while conducting due diligence of franchisees before appointing them", "franchisee due diligence checks"),
    (r"criteria for selection of centres", "selection criteria for centres"),
    (r"guidelines on provision for training of franchisees", "franchisee training guidelines"),
    (r"guidelines in respect of reporting, audit and inspection of franchisees", "franchisee reporting and audit"),
    (r"authorised dealers category-ii issue forex pre-paid cards", "AD Category-II forex prepaid cards"),
    (r"forex pre-paid cards be used at duty free shops located at international airports in india", "forex prepaid cards at Duty Free"),
    (r"position of papua new guinea paper banknotes", "Papua New Guinea banknotes rule"),
    (r"position of old generation ‘1000 shillings \(ksh\)’ banknotes in kenya", "Kenya 1000 shillings banknote rule"),
    (r"national electronic funds transfer system", "NEFT system"),
    (r"advantages of using neft system", "NEFT benefits"),
    (r"neft system operate", "NEFT working process"),
    (r"indian financial system code", "IFSC code"),
    (r"find the ifsc of a bank-branch", "find bank IFSC"),
    (r"avail neft system for fund transfer / receipt", "NEFT transfer eligibility"),
    (r"limit on funds / amount to be remitted through neft system", "NEFT transfer limits"),
    (r"neft system be used for remitting funds even by those who do not have a bank account", "NEFT without bank account"),
    (r"send funds to my relative / friend residing abroad through neft system", "NEFT transfer abroad"),
    (r"operating hours of neft", "NEFT operational timings"),
    (r"essential details required for remitting funds through neft system", "NEFT transfer required details"),
    (r"customer charges levied by bank for neft transactions", "NEFT customer charges"),
    (r"use neft to transfer funds from / to nre and nro accounts", "NEFT for NRE NRO accounts"),
    (r"originate a neft transaction to draw / receive funds from another account", "NEFT inward draw transaction"),
    (r"track status of neft transactions initiated", "NEFT transaction tracking"),
    (r"help desk / contact point at the rbi", "RBI helpdesk contact"),
    (r"time should i expect for receipt of funds by beneficiary", "NEFT settlement time"),
    (r"penalties / compensation for delayed credit or return of funds by beneficiary bank", "NEFT delay penalty compensation"),
    (r"file complaint under the rb-ios 2021", "RB-IOS complaint filing"),
    (r"practices which should be adopted by a customer for safe and secure digital payments", "safe digital payment practices"),
    (r"legal entity identifier and how is it relevant in case of large value neft transactions", "LEI in large NEFT transfers"),
    (r"rbi's role with regard to conduct of government's banking transaction", "RBI government banking role"),
    (r"reserve bank of india discharge its statutory obligation of being 'banker to government'", "RBI banker to government role"),
    (r"receipted challan for payment made into government account made available", "government payment receipt challan"),
    (r"paper token is misplaced / lost", "lost paper token procedure"),
    (r"receipted challan is misplaced", "lost receipted challan procedure"),
    (r"agency banks compensated for conduct of central/state government banking", "agency bank government compensation"),
    (r"rbi’s role in goods and service tax regime", "RBI role in GST"),
    (r"maximum deposit amount insured by the dicgc", "DICGC insurance maximum limit"),
    (r"you know whether your bank is insured by the dicgc or not", "check if bank is DICGC insured"),
    (r"ceiling on amount of insured deposits kept by one person in different branches of a bank", "DICGC insurance across branches"),
    (r"deposit insurance be increased by depositing funds into several different accounts all at the same bank", "insurance increase via multiple accounts"),
    (r"deposits in different banks separately insured", "separate insurance for different banks"),
    (r"meaning of deposits held in the same capacity and same right; and deposits held in different capacity and different right", "deposits in same vs different capacity"),
    (r"bank deduct the amount of dues payable by the depositor", "bank deduction of depositor dues"),
    (r"pays the cost of deposits insurance", "deposit insurance cost payment"),
    (r"dicgc liable to pay", "DICGC payout conditions"),
    (r"insured bank withdraw from the dicgc coverage", "bank withdrawal from DICGC"),
    (r"dicgc withdraw deposit insurance coverage from any bank", "DICGC coverage cancellation"),
    (r"payments under rtgs final and irrevocable", "RTGS payment finality"),
    (r"benefits of using rtgs", "RTGS benefits"),
    (r"processing of rtgs different from that of national electronic funds transfer \(neft\) system", "RTGS vs NEFT processing"),
    (r"rtgs a 24x7 system or are there some timings applicable", "RTGS 24x7 availability"),
    (r"minimum / maximum amount stipulation for rtgs transactions", "RTGS minimum maximum limits"),
    (r"essential information that the remitting customer needs to furnish to the bank for making a remittance", "RTGS remittance required details"),
    (r"time taken for effecting funds transfer from one account to another through rtgs", "RTGS transfer duration"),
    (r"remitting customer initiate a transaction for a future date", "future date RTGS transfer"),
    (r"transaction be originated to draw \(receive\) funds from another account", "originate transaction to draw funds"),
    (r"rtgs transaction be tracked", "track RTGS transaction"),
    (r"customer eligible to get compensation for delay in returning the payment", "RTGS delay return compensation"),
    (r"utr number", "UTR number"),
    (r"legal entity identifier and how is it relevant in case of rtgs transactions", "LEI in RTGS transactions"),
    (r"authorized dealer \(ad\)", "Authorized Dealer AD"),
    (r"authorized by the reserve bank to sell foreign exchange for travel purposes", "entities authorized to sell forex"),
    (r"foreign currency can be carried in cash for travel abroad", "cash forex limits for travel"),
    (r"indian currency can be brought in while coming into india", "rupee currency limits entering India"),
    (r"foreign exchange can be brought in while visiting india", "forex import limits into India"),
    (r"pay by cash full rupee equivalent of foreign exchange being purchased for travel abroad", "cash payment for forex purchase"),
    (r"time-frame for a traveller who has returned to india to surrender foreign exchange", "forex surrender time limit on return"),
    (r"foreign coins be surrendered to an authorised dealer on return from abroad", "surrender of foreign coins"),
    (r"category of visit which requires prior approval from the reserve bank or the government of india", "travel categories requiring RBI approval"),
    (r"permission is required for receiving grant/donation from abroad under the foreign contribution regulation act, 1976", "FCRA grant donation permission"),
    (r"permitted to hold international credit card \(icc\) and international debit card \(idc\) for undertaking foreign exchange transactions", "eligibility for ICC and IDC cards"),
    (r"jewellery can be carried while going abroad", "jewellery export allowance abroad"),
    (r"resident extend local hospitality to a non-resident", "local hospitality to non-residents"),
    (r"residents purchase air tickets in india for their travel not touching india", "air ticket booking for travel outside India"),
    (r"meeting of medical expenses of a nri close relative, in india, by resident individuals permitted", "paying medical expenses of NRI relative"),
    (r"person resident in india hold assets outside india", "holding foreign assets by residents"),
    (r"indo-nepal remittance facility scheme under neft ecosystem", "Indo-Nepal remittance scheme"),
    (r"salient features of inrf scheme", "INRF scheme features"),
    (r"limit on number of such transactions", "INRF transaction count limits"),
    (r"documents required to be presented by the remitter", "INRF remitter required documents"),
    (r"transactions flow from india to nepal and what are the timelines for completion of the transactions", "India-Nepal transfer workflow and timeline"),
    (r"remitting customer in india know about the branches of nsbl and the outlets of prabhu money transfer", "NSBL and Prabhu Money Transfer outlets"),
    (r"remitter get back money if it is not delivered to the beneficiary", "INRF failed transfer refund"),
    (r"charges for availing the remittance facility", "Indo-Nepal remittance charges"),
    (r"be contacted for redressal of grievances under the scheme", "INRF grievance redressal contact"),
    (r"bi-lateral arrangements between bank / non-banks in india and those in nepal covered under inrf scheme", "bilateral arrangements under INRF"),
    (r"joint account can be continued for family pension after death of a pensioner", "joint account continuation for family pension"),
    (r"pension credited to the pensioner's account by the paying branch", "pension credit timing"),
    (r"pension paying bank recover the excess amount credited to the pensioner’s account", "recovery of excess pension credit"),
    (r"acknowledgement be given by pension paying banks while accepting life certificates from pensioners", "life certificate acknowledgement receipt"),
    (r"it possible to submit life certificates without visiting the branch", "digital life certificate submission"),
    (r"pensioner withdraw pension from his/ her account when he/she is not able to sign or put thumb/toe impression or unable to be present in the bank", "pension withdrawal for incapacitated pensioner"),
    (r"pensioner is entitled for any compensation from the abs for delayed credit of pension/ arrears of pension", "delayed pension compensation"),
    (r"foreign currency account", "foreign currency account"),
    (r"resident continue to maintain an account outside india which was opened by him when he was a non-resident", "retaining foreign account after returning to India"),
    (r"status of the account held outside india on the demise of the account holder", "foreign account status after holder demise"),
    (r"payment and settlement systems act, 2007 \(pss act, 2007\) came into effect", "PSS Act 2007 effective date"),
    (r"objective of the pss act, 2007", "PSS Act objective"),
    (r"regulations made under the pss act, 2007 and when did they come into force", "PSS Act regulations and enforcement"),
    (r"objectives of these two regulations", "PSS regulations objectives"),
    (r"“payment obligation”", "payment obligation definition"),
    (r"“payment instruction”", "payment instruction definition"),
    (r"“settlement”", "settlement meaning in PSS"),
    (r"“payment system” under the pss act, 2007", "payment system definition under PSS Act"),
    (r"entities operating a payment system or intending to operate a payment system required to get a license, approval or authorization for the purpose", "payment system authorization license mandate"),
    (r"application fee to be submitted along with the application for authorization", "payment system authorization fee"),
    (r"foreign entities allowed to operate a payment system in india", "foreign entities operating payment systems"),
    (r"foreign entities required to get a license or approval or authorization from reserve bank before commencing operations", "foreign payment entity RBI license requirement"),
    (r"financial market infrastructures", "Financial Market Infrastructures FMI"),
    (r"foreign financial market infrastructure \(fmi\) start operations in india", "foreign FMI operations in India"),
    (r"services which a foreign entity can provide", "services permitted for foreign entities"),
    (r"factors which the reserve bank will consider while deciding on an application submitted for authorization", "RBI payment authorization evaluation factors"),
    (r"parameters taken into consideration for giving authorisation to the applicants", "parameters for payment system authorization"),
    (r"reserve bank refuse to grant authorization to commence or operate a payment system", "RBI refusal of payment authorization"),
    (r"reserve bank revoke authorization granted under the pss act 2007", "RBI revocation of payment authorization"),
    (r"appellate authority to whom an aggrieved applicant whose application for authorization is refused or a system provider whose authorization is revoked, can appeal", "appellate authority under PSS Act"),
    (r"reserve bank collect any authorisation fees and direct the applicant to furnish a security deposit", "RBI authorization fee and security deposit"),
    (r"reserve bank can call for returns, information etc., from the system provider with regard to the operation of the payment system", "RBI power to call for payment returns"),
    (r"reserve bank can share such information as received above with other regulators, etc", "RBI sharing payment info with regulators"),
    (r"reserve bank inspect the premises of the system provider", "RBI inspection of payment provider premises"),
    (r"reserve bank conduct inspection of foreign entities authorized by it but located in foreign jurisdictions", "RBI inspection of overseas entities"),
    (r"reserve bank issue directions to the system provider", "RBI directions to payment providers"),
    (r"duties of a system provider under the pss act, 2007", "duties of payment system provider"),
    (r"mechanism for settlement of disputes under the pss act, 2007", "dispute settlement under PSS Act"),
    (r"consequences of dishonour of electronic fund transfer under the pss act, 2007", "electronic fund transfer dishonour consequences"),
    (r"penalties or punitive action laid down under the pss act, 2007", "penalties under PSS Act"),
    (r"automated teller machine \(atm\)", "Automated Teller Machine ATM"),
    (r"white label atms \(wlas\)", "White Label ATMs WLA"),
    (r"services / facilities available at atms / wlas", "services available at ATMs and WLAs"),
    (r"pre-requisites for transacting at an atm / wla", "ATM transaction pre-requisites"),
    (r"personal identification number \(pin\)", "PIN number"),
    (r"cards issued by a bank in india be used at any atm / wla in the country", "using debit card at any ATM"),
    (r"on-us and off-us transaction", "On-Us vs Off-Us transactions"),
    (r"customers entitled to any free transactions at atms", "free ATM transactions quota"),
    (r"bank offer more number of free transactions at atms", "extra free ATM transactions by banks"),
    (r"above prescription of free transactions applicable to a basic savings bank deposit account \(bsbda\) also", "free ATM transactions for BSBDA accounts"),
    (r"type of transactions that are not to be counted under free transactions", "transactions excluded from free ATM count"),
    (r"know if the atm location is metro or non-metro", "identify metro vs non-metro ATM"),
    (r"customers charged for transactions at atms", "ATM transaction charges"),
    (r"charges prescribed by rbi for use of credit cards at atms and for withdrawal at atm located abroad", "credit card ATM withdrawal charges"),
    (r"time limit for the card issuing bank to recredit the customer’s account for a failed atm / wla transaction indicated under q. no. 18", "failed ATM transaction refund turnaround time"),
    (r"customers eligible for compensation for delays beyond days of a failed transaction", "failed ATM transaction delay compensation"),
    (r"course of action for the customer if the reversal and compensation are not carried by the bank", "remedy for non-reversal of failed ATM transaction"),
    (r"magnetic stripe cards and emv chip & pin cards", "magnetic stripe vs EMV chip cards"),
    (r"banks for issuing magnetic stripe cards or emv chip & pin cards", "mandate for EMV chip card issuance"),
    (r"charges levied for the collection of these usd denominated cheques", "USD cheque collection charges"),
    (r"us regulations applicable to usd cheque collection", "US regulations on USD cheques"),
    (r"customer choose the mode for collecting usd cheques", "choice of USD cheque collection mode"),
    (r"customers given credit and allowed to use the funds after sight of credit in the nostro accounts of banks", "credit availability on Nostro sight"),
    (r"instructions for facilitating customer awareness and redressing customer complaints", "customer awareness and complaints rules"),
    (r"other instructions to banks in this regard", "additional instructions for banks"),
    (r"government security \(g-sec\)", "Government Security G-Sec"),
    (r"g-secs issued", "G-Sec issuance process"),
    (r"different types of auctions used for issue of securities", "security auction types"),
    (r"open market operations \(omos\)", "Open Market Operations OMO"),
    (r"liquidity adjustment facility \(laf\) and whether re-repo in government securities market is allowed", "LAF facility and re-repo rules"),
    (r"trading in g-secs take place and what regulations are applicable to prevent abuse", "G-Sec trading process and market abuse rules"),
    (r"major players in the g-secs market", "key participants in G-Sec market"),
    (r"do's and don’ts prescribed by rbi for the co-operative banks dealing in g-secs", "G-Sec guidelines for co-operative banks"),
    (r"dealing transactions recorded by the dealing desk", "recording dealing desk transactions"),
    (r"important considerations while undertaking security transactions", "security transaction considerations"),
    (r"get information about the price of a g-sec", "tracking G-Sec market price"),
    (r"g-secs transactions reported", "reporting G-Sec trades"),
    (r"delivery versus payment \(dvp\) settlement", "Delivery vs Payment DvP settlement"),
    (r"role of the clearing corporation of india limited \(ccil\)", "CCIL role and functions"),
    (r"‘when issued’ market and “short sale”", "When Issued market and short selling"),
    (r"basic mathematical concepts one should know for calculations involved in bond prices and yields", "math concepts for bond price and yield"),
    (r"price of a bond calculated", "bond price calculation formula"),
    (r"relationship between yield and price of a bond", "bond yield vs price relationship"),
    (r"yield of a bond calculated", "calculating bond yield"),
    (r"day count conventions used in calculating bond yields", "bond yield day count conventions"),
    (r"yield of a t- bill calculated", "T-Bill yield calculation"),
    (r"important guidelines for valuation of securities", "security valuation guidelines"),
    (r"risks involved in holding g-secs", "risks in holding G-Secs"),
    (r"different money market instruments", "money market instruments"),
    (r"role and functions of fimmda & fbil", "FIMMDA and FBIL roles"),
    (r"various websites that give information on g-secs", "websites for G-Sec info"),
]

def paraphrase_topic(topic):
    t_clean = topic.strip().lower()
    for pat, rep in PHRASE_SIMPLIFICATIONS:
        if re.search(pat, t_clean):
            return rep
    
    # Generic clause trimming
    t_clean = re.sub(r"\s+(?:and\s+what\s+are\s+its\s+benefits|and\s+for\s+what\s+purpose|in\s+compliance\s+with.*|under\s+section.*|under\s+the\s+.*act.*|while\s+visiting\s+india.*|while\s+going\s+abroad.*|for\s+travel\s+abroad.*|located\s+at\s+international\s+airports.*|in\s+india|outside\s+india)\b.*$", "", t_clean)
    t_clean = re.sub(r"\s+(?:can\s+be\s+.*|is\s+permitted.*|is\s+mandatory.*|is\s+required.*|are\s+applicable.*|take\s+place.*|be\s+made.*|be\s+given.*|be\s+used.*|be\s+opened.*|be\s+credited.*|be\s+surrendered.*|be\s+recovered.*|be\s+issued.*|be\s+held.*)\b.*$", "", t_clean)
    words = t_clean.split()
    if len(words) > 4:
        return " ".join(words[:4])
    return t_clean.strip()

def build_paraphrased_query(q_raw):
    q = strip_numbering(q_raw).rstrip('?.').strip()
    ql = q.lower()

    # Aggressive Hindi vocabulary mapping to remove exact lexical matches
    vocab_map = {
        r'\bdeposit(s)?\b': 'jama',
        r'\baccount(s)?\b': 'khata',
        r'\binterest( rate)?\b': 'byaaj',
        r'\btransfer(red|s)?\b': 'bhejna',
        r'\bloan(s)?\b': 'karz',
        r'\bdocument(s)?\b': 'kagzaat',
        r'\bpayment(s)?\b': 'bhugtan',
        r'\brules?\b': 'niyam',
        r'\blimits?\b': 'seema',
        r'\bamount\b': 'rakam',
        r'\bapplication\b': 'form',
        r'\bapply\b': 'aavedan',
        r'\bfunds?\b': 'paise',
        r'\bbank\b': 'bank',
        r'\bpenalty\b': 'jurmana',
        r'\bcompensation\b': 'muawza',
        r'\bcomplaint\b': 'shikayat',
        r'\bforeign\b': 'videshi',
        r'\bcurrency\b': 'mudra',
        r'\bsecurity\b': 'suraksha',
        r'\bbranch(es)?\b': 'shakha',
        r'\bcheque(s)?\b': 'check',
        r'\bwithdraw(al)?\b': 'nikalna',
        r'\breceipt(s)?\b': 'raseed',
        r'\bpension(er)?\b': 'pension',
        r'\bdeath\b': 'maut',
        r'\bissue(d)?\b': 'jaari',
        r'\bfee(s)?\b': 'fees',
        r'\bcard(s)?\b': 'card'
    }

    if re.match(r'how much (of )?(.*)', ql):
        m = re.match(r'how much (of )?(.*)', ql)
        top = paraphrase_topic(m.group(2))
        return f"{top} ki maximum seema aur charges kitne hote hain"

    if re.match(r'how many (.*)', ql):
        m = re.match(r'how many (.*)', ql)
        top = paraphrase_topic(m.group(1))
        return f"{top} ki total ginti aur seema kitni hai"

    if re.match(r'how (will|does|do|can|is|are) (.*)', ql):
        m = re.match(r'how (will|does|do|can|is|are) (.*)', ql)
        top = paraphrase_topic(m.group(2))
        return f"{top} ka tareeqa aur step-by-step procedure kya hota hai"

    if re.match(r'(whether|can|is there|are there|should|is|are) (.*)', ql):
        m = re.match(r'(whether|can|is there|are there|should|is|are) (.*)', ql)
        top = paraphrase_topic(m.group(2))
        return f"kya {top} allow kiya jata hai ya koi khaas rukawat hai"

    if re.match(r'(what if|when should|when did|when) (.*)', ql):
        m = re.match(r'(what if|when should|when did|when) (.*)', ql)
        top = paraphrase_topic(m.group(2))
        return f"agar {top} toh aise case mein kya niyam lagta hai"

    if re.match(r'what is the (procedure|process|mandate|time ?frame) for (.*)', ql):
        m = re.match(r'what is the (procedure|process|mandate|time ?frame) for (.*)', ql)
        kind, top = m.group(1), paraphrase_topic(m.group(2))
        return f"{top} ke niyam aur tareeqa kya hain"

    if re.match(r'what (is|are) (meant by )?(.*)', ql):
        m = re.match(r'what (is|are) (meant by )?(.*)', ql)
        top = paraphrase_topic(m.group(3))
        return f"{top} ka arth aur details kya hain"

    if re.match(r'who (is|are) (a |an )?(.*)', ql):
        m = re.match(r'who (is|are) (a |an )?(.*)', ql)
        top = paraphrase_topic(m.group(3))
        return f"{top} ki pehchaan aur role kya hai"

    if re.match(r'what does (.*?) (mean|insure|cover)', ql):
        m = re.match(r'what does (.*?) (mean|insure|cover)', ql)
        top, verb = paraphrase_topic(m.group(1)), m.group(2)
        return f"{top} kis cheez ko {verb} karta hai"

    if re.match(r'who can (.*)', ql):
        m = re.match(r'who can (.*)', ql)
        top = paraphrase_topic(m.group(1))
        return f"{top} ke liye yogyata aur criteria kya hoti hai"

    if re.match(r'who (.*)', ql):
        m = re.match(r'who (.*)', ql)
        top = paraphrase_topic(m.group(1))
        return f"{top} kaun hota hai aur niyam kya hain"

    top = paraphrase_topic(q)
    base_query = f"{top} ke baare mein niyam aur jaankari chahiye"
    
    # Apply aggressive vocab replacement
    for eng, hin in vocab_map.items():
        base_query = re.sub(eng, hin, base_query, flags=re.IGNORECASE)
        
    return base_query

def regenerate_all_queries():
    # Load existing queries_v3_final.json to preserve pilot queries Q01-Q75
    in_file = ROOT / "data" / "processed" / "queries_v3_final.json"
    existing = json.load(open(in_file, encoding='utf-8'))
    
    pilot_queries = existing[:75]
    auto_queries_old = existing[75:]
    
    updated_auto = []
    for q_obj in auto_queries_old:
        s_q = q_obj.get("source_question", "")
        new_text = build_paraphrased_query(s_q)
        
        updated_item = dict(q_obj)
        updated_item["text"] = new_text
        updated_item["review_status"] = "auto_generated_v3_paraphrased"
        updated_auto.append(updated_item)
        
    full_v3 = pilot_queries + updated_auto
    
    with open(in_file, "w", encoding="utf-8") as f:
        json.dump(full_v3, f, indent=2, ensure_ascii=False)
        
    print(f"Regenerated and saved {len(full_v3)} queries to {in_file}")
    return full_v3

if __name__ == "__main__":
    regenerate_all_queries()
