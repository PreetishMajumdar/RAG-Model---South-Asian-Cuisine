"""
Phase B.1.1 & B.1.2: Data Ingestion and Chunking (BASELINE MODEL)
Responsibility: Dinghua
Description: Scrapes Wikipedia, Wikibooks, and Blogs, applying the Baseline 
800-character flat chunking strategy.
"""
import requests
from bs4 import BeautifulSoup
import re
import json
import time
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Configuration
HEADERS = {"User-Agent": "COMP64702_RAG_Project_Team PythonRequests"}

# --- BASELINE CHUNKING STRATEGY ---
# Fixed 800 characters, 100 character overlap
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100,
    length_function=len,
    separators=["\n\n", "\n", ". ", " "]
)

def baseline_chunker(clean_text, base_metadata, is_table=False):
    """Splits text into flat 800-char chunks with basic index tracking."""
    chunks = []
    # Don't split tables or short text
    if is_table or len(clean_text) <= 800:
        metadata = base_metadata.copy()
        metadata["chunk_index"] = 1
        metadata["total_chunks"] = 1
        chunks.append({"text": clean_text, "metadata": metadata})
    else:
        split_texts = text_splitter.split_text(clean_text)
        total = len(split_texts)
        for i, chunk in enumerate(split_texts):
            metadata = base_metadata.copy()
            metadata["chunk_index"] = i + 1
            metadata["total_chunks"] = total
            chunks.append({"text": chunk, "metadata": metadata})
            
    return chunks


# --- 1. WIKIBOOKS SCRAPER ---
def scrape_wikibooks(urls):
    page_data = []
    for url in urls:
        print(f"Scraping Wikibooks: {url}")
        response = requests.get(url, headers=HEADERS)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, "html.parser") 
            title_element = soup.find("h1", id="firstHeading")
            recipe_name = title_element.text.replace("Cookbook:", "").strip() if title_element else "Unknown"
            content_area = soup.find(id="mw-content-text")
            
            if content_area:
                elements = content_area.find_all(['h2', 'h3', 'p', 'ul', 'ol', 'table'])
                current_section = "Introduction"
                
                for element in elements:
                    if element.name in ['h2', 'h3']:
                        current_section = element.get_text().replace('[edit]', '').strip()
                    
                    elif element.name == 'p':
                        text = element.get_text().strip()
                        if "Cookbook |" in text or "Recipes |" in text: continue 
                        if len(text) > 15:
                            clean_text = re.sub(r'\[\w+\]', '', text)
                            meta = {"source": "Wikibooks", "topic": recipe_name, "section": current_section, "url": url}
                            page_data.extend(baseline_chunker(clean_text, meta, is_table=False))
                    
                    elif element.name in ['ul', 'ol']:
                        if element.parent.has_attr('class') and 'navbox' in element.parent['class']: continue
                        list_items = element.find_all('li')
                        if list_items:
                            combined_list_text = "\n".join([f"- {li.get_text().strip()}" for li in list_items if li.get_text().strip()])
                            if len(combined_list_text) > 10:
                                clean_text = re.sub(r'\[\w+\]', '', combined_list_text)
                                meta = {"source": "Wikibooks", "topic": recipe_name, "section": current_section, "url": url}
                                page_data.extend(baseline_chunker(clean_text, meta, is_table=False))
                                
                    elif element.name == 'table':
                        if element.has_attr('class') and any(c in ['navbox', 'toc', 'infobox'] for c in element['class']): continue
                        table_rows = []
                        for row in element.find_all('tr'):
                            cells = [cell.get_text(strip=True) for cell in row.find_all(['th', 'td'])]
                            if any(cells): table_rows.append(" | ".join(cells))
                        if table_rows:
                            combined_table_text = "Table Data:\n" + "\n".join(table_rows)
                            clean_text = re.sub(r'\[\w+\]', '', combined_table_text)
                            meta = {"source": "Wikibooks", "topic": recipe_name, "section": current_section, "url": url}
                            page_data.extend(baseline_chunker(clean_text, meta, is_table=True))
        time.sleep(1)
    return page_data


# --- 2. BLOG SCRAPER ---
def scrape_blog(urls):
    page_data = []
    for url in urls:
        print(f"Scraping Blog: {url}")
        response = requests.get(url, headers=HEADERS)
        if response.status_code == 200:
            soup = BeautifulSoup(response.content, "html.parser")
            title_element = soup.find("h1", class_="entry-title") or soup.find("h1")
            post_title = title_element.get_text().strip() if title_element else "Unknown"
            content_area = soup.find("div", class_="entry-content")

            if content_area:
                elements = content_area.find_all(['h2', 'h3', 'p', 'ul'])
                current_section = "Blog Narrative"

                for element in elements:
                    if element.name in ['h2', 'h3']:
                        current_section = element.get_text().strip()
                    elif element.name == 'p':
                        text = element.get_text().strip()
                        if "Share this:" in text or "Like this:" in text or len(text) < 15: continue
                        clean_text = re.sub(r'\[\w+\]', '', text).strip()
                        if clean_text:
                            meta = {"source": "Around the World in 80 Cuisines", "topic": post_title, "section": current_section, "url": url}
                            page_data.extend(baseline_chunker(clean_text, meta))
                    elif element.name == 'ul':
                        list_items = [li.get_text().strip() for li in element.find_all('li') if "Share" not in li.get_text() and li.get_text().strip()]
                        if list_items:
                            combined_list_text = "\n".join([f"- {item}" for item in list_items])
                            if len(combined_list_text) > 10:
                                meta = {"source": "Around the World in 80 Cuisines", "topic": post_title, "section": current_section, "url": url}
                                page_data.extend(baseline_chunker(combined_list_text, meta))
        time.sleep(1)
    return page_data


# --- 3. WIKIPEDIA SCRAPER & HELPERS ---
def clean_wiki_text(text):
    cleaned = re.sub(r'\[[0-9a-zA-Z]+\]', '', text).replace('[edit]', '')
    cleaned = re.sub(r'\s+', ' ', cleaned)
    return re.sub(r'\[\d+(,\s*\d+)*\]', '', cleaned).strip()

def is_valid_element(element):
    invalid_classes = ['infobox', 'navbox', 'toc', 'reflist', 'reference', 'metadata', 'sidebar', 'mw-empty-elt', 'catlinks', 'printfooter']
    for parent in [element] + list(element.parents):
        classes = parent.get('class')
        if classes and any(c in invalid_classes for c in (classes if isinstance(classes, list) else [classes])): return False
    return True

def wiki_table_to_markdown(table_tag):
    rows = table_tag.find_all('tr')
    md_lines, header_written, has_real_content = [], False, False
    for row in rows:
        cells = row.find_all(['th', 'td'], recursive=False)
        if not cells: continue
        row_data = [clean_wiki_text(c.get_text(separator=" ")).replace('\n', ' ') for c in cells]
        if not any(cell.strip() for cell in row_data): continue
        has_real_content = True
        md_lines.append("| " + " | ".join(row_data) + " |")
        if row.find('th') and not header_written:
            md_lines.append("|" + "|".join(["---"] * len(row_data)) + "|")
            header_written = True
    if not has_real_content: return ""
    if md_lines and not header_written:
        md_lines.insert(1, "|" + "|".join(["---"] * (len(md_lines[0].split('|')) - 2)) + "|")
    return "\n".join(md_lines)

def wiki_list_to_markdown(list_tag, level=0):
    items = []
    for li in list_tag.find_all('li', recursive=False):
        text_parts = [child.get_text(separator=" ") if child.name else child.string for child in li.children if child.name not in ['ul', 'ol'] and child]
        li_text = clean_wiki_text(" ".join(filter(None, text_parts)))
        if li_text: items.append(f"{'  ' * level}- {li_text}")
        for nested_list in li.find_all(['ul', 'ol'], recursive=False):
            items.append(wiki_list_to_markdown(nested_list, level + 1))
    return "\n".join(items)

def scrape_wikipedia(urls):
    corpus_data = []
    session = requests.Session()
    session.headers.update(HEADERS)
    skip_sections = ["see also", "references", "further reading", "external links", "bibliography", "notes"]

    for i, url in enumerate(urls):
        print(f"[{i+1}/{len(urls)}] Scraping Wikipedia: {url}")
        try:
            response = session.get(url, timeout=15)
            response.raise_for_status()
        except Exception as e:
            continue
            
        soup = BeautifulSoup(response.content, 'html.parser')
        for tag in soup.find_all(['sup', 'span', 'div', 'table', 'ul', 'figure', 'figcaption'], class_=re.compile(r'(reference|mw-editsection|noprint|thumb|gallery)')):
            tag.decompose()
            
        title_tag = soup.find('h1', id='firstHeading')
        topic = title_tag.get_text(strip=True) if title_tag else "Unknown Topic"
        content_div = soup.find('div', class_='mw-parser-output')
        if not content_div: continue

        current_section = "Summary"
        skip_current_section = False

        for element in content_div.find_all(['h2', 'h3', 'h4', 'h5', 'p', 'ul', 'ol', 'table']):
            if not is_valid_element(element): continue

            if element.name in ['h2', 'h3', 'h4', 'h5']:
                current_section = clean_wiki_text(element.get_text(separator=" "))
                skip_current_section = any(skip in current_section.lower() for skip in skip_sections)
                continue
                
            if skip_current_section or any(parent.name in ['table', 'ul', 'ol'] for parent in element.parents): continue
                
            base_metadata = {"source": "Wikipedia", "topic": topic, "section": current_section, "url": url}

            if element.name == 'p':
                text = clean_wiki_text(element.get_text(separator=" "))
                if len(text) > 15 and re.search(r'[a-zA-Z0-9]', text):
                    corpus_data.extend(baseline_chunker(text, base_metadata))
            elif element.name in ['ul', 'ol']:
                md_list = wiki_list_to_markdown(element)
                if len(md_list) > 10 and re.search(r'[a-zA-Z0-9]', md_list):
                    corpus_data.extend(baseline_chunker(md_list, base_metadata))
            elif element.name == 'table':
                md_table = wiki_table_to_markdown(element)
                if md_table and re.search(r'[a-zA-Z0-9]', md_table):
                    corpus_data.extend(baseline_chunker(md_table, base_metadata, is_table=True))
        time.sleep(1)
    return corpus_data


# --- MAIN EXECUTION ---
if __name__ == "__main__":
    # The full URL lists provided in your code
    wiki_urls = [
        "https://en.wikipedia.org/wiki/Aloo_gobi", "https://en.wikipedia.org/wiki/Andhra_cuisine",
        "https://en.wikipedia.org/wiki/Anglo-Indian_cuisine", "https://en.wikipedia.org/wiki/Assamese_cuisine",
        "https://en.wikipedia.org/wiki/Awadhi_cuisine", "https://en.wikipedia.org/wiki/Balochi_cuisine",
        "https://en.wikipedia.org/wiki/Bangladeshi_cuisine", "https://en.wikipedia.org/wiki/Bengali_cuisine",
        "https://en.wikipedia.org/wiki/Bhojpuri_cuisine", "https://en.wikipedia.org/wiki/Bihari_cuisine",
        "https://en.wikipedia.org/wiki/Chettinad_cuisine", "https://en.wikipedia.org/wiki/Chutney",
        "https://en.wikipedia.org/wiki/Cuisine_of_Uttar_Pradesh", "https://en.wikipedia.org/wiki/Dal",
        "https://en.wikipedia.org/wiki/Dosa_(food)", "https://en.wikipedia.org/wiki/East_India#Cuisine",
        "https://en.wikipedia.org/wiki/Goan_Catholic_cuisine", "https://en.wikipedia.org/wiki/Goan_cuisine",
        "https://en.wikipedia.org/wiki/Gujarati_cuisine", "https://en.wikipedia.org/wiki/Hazara_cuisine",
        "https://en.wikipedia.org/wiki/Hyderabadi_cuisine", "https://en.wikipedia.org/wiki/Indian_Chinese_cuisine",
        "https://en.wikipedia.org/wiki/Indian_South_Africans#Cuisine", "https://en.wikipedia.org/wiki/Indian_cuisine",
        "https://en.wikipedia.org/wiki/Indian_fast_food", "https://en.wikipedia.org/wiki/Jain_vegetarianism",
        "https://en.wikipedia.org/wiki/Jharkhandi_cuisine", "https://en.wikipedia.org/wiki/Karahi",
        "https://en.wikipedia.org/wiki/Karnataka_cuisine", "https://en.wikipedia.org/wiki/Kashmiri_cuisine",
        "https://en.wikipedia.org/wiki/Kerala_cuisine", "https://en.wikipedia.org/wiki/Kodava_people",
        "https://en.wikipedia.org/wiki/Lahori_cuisine", "https://en.wikipedia.org/wiki/Maharashtrian_cuisine",
        "https://en.wikipedia.org/wiki/Maithil_cuisine", "https://en.wikipedia.org/wiki/Malaysian_Indian_cuisine",
        "https://en.wikipedia.org/wiki/Maldivian_cuisine", "https://en.wikipedia.org/wiki/Malvani_cuisine",
        "https://en.wikipedia.org/wiki/Mangalorean_Catholic_Cuisine", "https://en.wikipedia.org/wiki/Manipuri_cuisine",
        "https://en.wikipedia.org/wiki/Meitei_cuisine", "https://en.wikipedia.org/wiki/Mizo_cuisine",
        "https://en.wikipedia.org/wiki/Mughlai_cuisine", "https://en.wikipedia.org/wiki/Muhajir_cuisine",
        "https://en.wikipedia.org/wiki/Naan", "https://en.wikipedia.org/wiki/Naga_cuisine",
        "https://en.wikipedia.org/wiki/Nepalese_cuisine", "https://en.wikipedia.org/wiki/Newari_cuisine",
        "https://en.wikipedia.org/wiki/North_East_Indian_cuisine", "https://en.wikipedia.org/wiki/North_Indian_cuisine",
        "https://en.wikipedia.org/wiki/North_Karnataka", "https://en.wikipedia.org/wiki/Odia_cuisine",
        "https://en.wikipedia.org/wiki/Oriya_cuisine", "https://en.wikipedia.org/wiki/Pakistani_cuisine",
        "https://en.wikipedia.org/wiki/Parsi_cuisine", "https://en.wikipedia.org/wiki/Pashtun_cuisine",
        "https://en.wikipedia.org/wiki/Punjabi_cuisine", "https://en.wikipedia.org/wiki/Raita",
        "https://en.wikipedia.org/wiki/Rajasthani_cuisine", "https://en.wikipedia.org/wiki/Rice",
        "https://en.wikipedia.org/wiki/Sambar_(dish)", "https://en.wikipedia.org/wiki/Saraiki_cuisine",
        "https://en.wikipedia.org/wiki/Seekh_kebab", "https://en.wikipedia.org/wiki/Shahi_paneer",
        "https://en.wikipedia.org/wiki/Sikkimese_cuisine", "https://en.wikipedia.org/wiki/Sindhi_cuisine",
        "https://en.wikipedia.org/wiki/South_Asian_cuisine", "https://en.wikipedia.org/wiki/South_Indian_cuisine",
        "https://en.wikipedia.org/wiki/South_Karnataka", "https://en.wikipedia.org/wiki/Sri_Lankan_cuisine",
        "https://en.wikipedia.org/wiki/Tamil_cuisine", "https://en.wikipedia.org/wiki/Telugu_cuisine",
        "https://en.wikipedia.org/wiki/Thali", "https://en.wikipedia.org/wiki/Tripuri_cuisine",
        "https://en.wikipedia.org/wiki/Udupi_cuisine"
    ]
    
    wikibooks_urls = [
        "https://en.wikibooks.org/wiki/Cookbook:Afghan_Bread", "https://en.wikibooks.org/wiki/Cookbook:Chicken_Tikka",
        "https://en.wikibooks.org/wiki/Cookbook:Naan", "https://en.wikibooks.org/wiki/Cookbook:Arisa_Pitha_(Fried_Indian_Sweet_Rice_Pastry)",
        "https://en.wikibooks.org/wiki/Cookbook:Chyapa_Shutki_Bharta", "https://en.wikibooks.org/wiki/Cookbook:Bhuna_Khichuri_(Bengali_Rice_and_Lentils)",
        "https://en.wikibooks.org/wiki/Cookbook:Mishti_Doi_(Bengali_Sweetened_Yogurt)", "https://en.wikibooks.org/wiki/Cookbook:Murghi_Korma_(Chicken_Korma)",
        "https://en.wikibooks.org/wiki/Cookbook:Pudina_Hilsa_(Bengali_Fish_with_Mint)", "https://en.wikibooks.org/wiki/Cookbook:Rosogulla_(Bengali_Milk_Balls_in_Syrup)",
        "https://en.wikibooks.org/wiki/Cookbook:Fried_Wheat_Bread_Balls_(Bhatoora)", "https://en.wikibooks.org/wiki/Cookbook:Makki_di_Roti_(Indian_Cornmeal_Flatbread)",
        "https://en.wikibooks.org/wiki/Cookbook:Potato_and_Cauliflower_Curry_(Aloo_Gobi)", "https://en.wikibooks.org/wiki/Cookbook:Salty_(Namkin)_Lassi",
        "https://en.wikibooks.org/wiki/Cookbook:Tandoori_Masala", "https://en.wikibooks.org/wiki/Cookbook:Appam_(Fermented_Rice_Pancake)",
        "https://en.wikibooks.org/wiki/Cookbook:Bonda_(South_Indian_Vegetable_Fritter)", "https://en.wikibooks.org/wiki/Cookbook:Hyderabad_Biryani",
        "https://en.wikibooks.org/wiki/Cookbook:Hyderabadi_Fried_Bread_with_Syrup_and_Nuts_(Double_ka_meetha)", "https://en.wikibooks.org/wiki/Cookbook:Idiyappam_(South_Indian_Rice_Noodles)",
        "https://en.wikibooks.org/wiki/Cookbook:Idli_(Steamed_Rice_and_Black_Gram_Bread)", "https://en.wikibooks.org/wiki/Cookbook:Kesari_(South_Indian_Semolina_Pudding)",
        "https://en.wikibooks.org/wiki/Cookbook:Khara_Pongal_(Rice_and_Mung_Bean_Porridge)", "https://en.wikibooks.org/wiki/Cookbook:Ragi_Dosa_(South_Indian_Millet_and_Rice_Pancake)",
        "https://en.wikibooks.org/wiki/Cookbook:Tamate_Ka_Kut_(Hyderabadi_Tomato_Curry)", "https://en.wikibooks.org/wiki/Cookbook:Chukauni_(Nepalese_Potato_Salad)",
        "https://en.wikibooks.org/wiki/Cookbook:Jhilinga_(Nepalese_Rice_Fritters)", "https://en.wikibooks.org/wiki/Cookbook:Masyaura_(Nepali_Fermented_Vegetable_Balls)",
        "https://en.wikibooks.org/wiki/Cookbook:Tibetan_Meat_Momos", "https://en.wikibooks.org/wiki/Cookbook:Aloo_Tikki_(Spiced_Potato_Patties)",
        "https://en.wikibooks.org/wiki/Cookbook:Basin_Ki_Kadi_(Sindhi_Chickpea_Flour_Curry)", "https://en.wikibooks.org/wiki/Cookbook:Chola_and_Roti",
        "https://en.wikibooks.org/wiki/Cookbook:Gobi_Bhagi_(Spiced_Cauliflower_Stew)", "https://en.wikibooks.org/wiki/Cookbook:Mustard_and_Curry_Leaf_Lassi",
        "https://en.wikibooks.org/wiki/Cookbook:Plain_Lassi", "https://en.wikibooks.org/wiki/Cookbook:Rajma_(Sindhi_Kidney_Bean_Curry)",
        "https://en.wikibooks.org/wiki/Cookbook:Sai_Bhaji_(Sindhi_Vegetable_Curry)", "https://en.wikibooks.org/wiki/Cookbook:Seviyan_Ji_Khirni_(Sindhi_Vermicelli_Pudding)",
        "https://en.wikibooks.org/wiki/Cookbook:Sindhi_Chickpea_Confection", "https://en.wikibooks.org/wiki/Cookbook:Sindhi_Fried_Potatoes",
        "https://en.wikibooks.org/wiki/Cookbook:Sindhi_Pulao", "https://en.wikibooks.org/wiki/Cookbook:Sindhi_Raitha",
        "https://en.wikibooks.org/wiki/Cookbook:Sindhi_Spiced_Fish", "https://en.wikibooks.org/wiki/Cookbook:Sindhi_Spiced_Moong_Dal",
        "https://en.wikibooks.org/wiki/Cookbook:Sindhi_Vegetable_Kofta", "https://en.wikibooks.org/wiki/Cookbook:Sylheti_Beef_Curry",
        "https://en.wikibooks.org/wiki/Cookbook:Sylheti_Biryani", "https://en.wikibooks.org/wiki/Cookbook:Sylheti_Doner_Kebab",
        "https://en.wikibooks.org/wiki/Cookbook:Sylheti_Fried_Rice", "https://en.wikibooks.org/wiki/Cookbook:Sylheti_Handesh",
        "https://en.wikibooks.org/wiki/Cookbook:Sylheti_Onion_and_Rice_Fritters", "https://en.wikibooks.org/wiki/Cookbook:Sylheti_Rice_Pudding",
        "https://en.wikibooks.org/wiki/Cookbook:Sylheti_Tangy_Curry", "https://en.wikibooks.org/wiki/Cookbook:Butter_Tea",
        "https://en.wikibooks.org/wiki/Cookbook:Corom_Chatni_(Mango_Chutney_with_Hot_Chillies)", "https://en.wikibooks.org/wiki/Cookbook:Dum_ka_Qimah_(Spiced_Minced_Meat)",
        "https://en.wikibooks.org/wiki/Cookbook:Khatti_Dal_(Spiced_Tamarind_Pigeon_Peas)", "https://en.wikibooks.org/wiki/Cookbook:Malpua_(South_Asian_Sweet_Pancake)",
        "https://en.wikibooks.org/wiki/Cookbook:Mango_Chutney_(Chunky)", "https://en.wikibooks.org/wiki/Cookbook:Mango_Chutney_(Smooth)",
        "https://en.wikibooks.org/wiki/Cookbook:Masala_Chai_II", "https://en.wikibooks.org/wiki/Cookbook:Mild_Salty_Lassi",
        "https://en.wikibooks.org/wiki/Cookbook:Papadam_(Black_Gram_Flatbread)", "https://en.wikibooks.org/wiki/Cookbook:Papaya_Lassi",
        "https://en.wikibooks.org/wiki/Cookbook:Papri_Chaat_(Crispy_Indian_Snack_with_Potato)", "https://en.wikibooks.org/wiki/Cookbook:Phulourie_(Split_Pea_Fritters)",
        "https://en.wikibooks.org/wiki/Cookbook:Prawn_Curry", "https://en.wikibooks.org/wiki/Cookbook:Qabuli_(Central_Asian_Rice_Pilaf)",
        "https://en.wikibooks.org/wiki/Cookbook:Sweet_Lassi", "https://en.wikibooks.org/wiki/Cookbook:Sweet_Mango_Lassi",
        "https://en.wikibooks.org/wiki/Cookbook:Watalappam_(Sri_Lankan_Coconut_Custard)"
    ]
    
    blog_urls = [
        "https://aroundtheworldin80cuisinesblog.wordpress.com/category/06-northern-india/",
        "https://aroundtheworldin80cuisinesblog.wordpress.com/category/38-afghanistan/",
        "https://aroundtheworldin80cuisinesblog.wordpress.com/category/44-southern-india/",
        "https://aroundtheworldin80cuisinesblog.wordpress.com/category/54-pakistan/",
        "https://aroundtheworldin80cuisinesblog.wordpress.com/category/20-sri-lanka-and-the-maldives/"
    ]
    
    print("--- Scraping Wikipedia (Baseline) ---")
    wiki_data = scrape_wikipedia(wiki_urls)
    with open("wikipedia_corpus_baseline.json", "w", encoding="utf-8") as f:
        json.dump(wiki_data, f, indent=4, ensure_ascii=False)
        
    print("\n--- Scraping Wikibooks (Baseline) ---")
    wikibooks_data = scrape_wikibooks(wikibooks_urls)
    with open("wikibooks_corpus_baseline.json", "w", encoding="utf-8") as f:
        json.dump(wikibooks_data, f, indent=4, ensure_ascii=False)
        
    print("\n--- Scraping Blog (Baseline) ---")
    blog_data = scrape_blog(blog_urls)
    with open("blog_corpus_baseline.json", "w", encoding="utf-8") as f:
        json.dump(blog_data, f, indent=4, ensure_ascii=False)
        
    print("\n✅ Baseline Ingestion Complete! Saved to 3 separate JSON files.")