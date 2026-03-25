"""
Phase B.1.1 & B.1.2: Data Ingestion and Chunking
Responsibility: Dinghua
Description: Scrapes Wikipedia, Wikibooks, and Blogs, applying a Hierarchical 
(Parent-Child) chunking strategy to preserve context for the generator.
"""
import requests
from bs4 import BeautifulSoup
import re
import json
import time
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Configuration
HEADERS = {"User-Agent": "COMP64702_RAG_Project_Team PythonRequests"}
PARENT_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=150)
CHILD_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=50)

def parent_child_chunker(clean_text, base_metadata, is_table=False):
    """Splits text into searchable children while retaining the parent context."""
    chunks = []
    parents = [clean_text] if (is_table or len(clean_text) <= 1500) else PARENT_SPLITTER.split_text(clean_text)
        
    for p_idx, parent_text in enumerate(parents):
        children = [parent_text] if (is_table or len(parent_text) <= 300) else CHILD_SPLITTER.split_text(parent_text)
        for c_idx, child_text in enumerate(children):
            metadata = base_metadata.copy()
            metadata["parent_id"] = f"{base_metadata['topic']}_{base_metadata['section']}_p{p_idx}"
            metadata["parent_text"] = parent_text
            metadata["chunk_index"] = c_idx + 1
            metadata["total_chunks"] = len(children)
            chunks.append({"text": child_text, "metadata": metadata})
    return chunks

# --- 1. WIKIBOOKS SCRAPER ---
def scrape_wikibooks(urls):
    data = []
    for url in urls:
        print(f"Scraping Wikibooks: {url}")
        res = requests.get(url, headers=HEADERS)
        if res.status_code == 200:
            soup = BeautifulSoup(res.content, "html.parser")
            title = soup.find("h1", id="firstHeading").text.replace("Cookbook:", "").strip()
            content = soup.find(id="mw-content-text")
            if not content: continue
            
            section = "Introduction"
            for el in content.find_all(['h2', 'h3', 'p', 'ul', 'table']):
                if el.name in ['h2', 'h3']: section = el.text.replace('[edit]', '').strip()
                elif el.name == 'p' and len(el.text) > 15:
                    meta = {"source": "Wikibooks", "topic": title, "section": section, "url": url}
                    data.extend(parent_child_chunker(re.sub(r'\[\w+\]', '', el.text.strip()), meta))
        time.sleep(1)
    return data

# --- 2. BLOG SCRAPER ---
def scrape_blog(urls):
    data = []
    for url in urls:
        print(f"Scraping Blog: {url}")
        res = requests.get(url, headers=HEADERS)
        if res.status_code == 200:
            soup = BeautifulSoup(res.content, "html.parser")
            title_el = soup.find("h1", class_="entry-title") or soup.find("h1")
            title = title_el.text.strip() if title_el else "Unknown"
            content = soup.find("div", class_="entry-content")
            if not content: continue
            
            section = "Narrative"
            for el in content.find_all(['h2', 'h3', 'p']):
                if el.name in ['h2', 'h3']: section = el.text.strip()
                elif el.name == 'p' and len(el.text) > 15 and "Share this:" not in el.text:
                    meta = {"source": "Around the World Blog", "topic": title, "section": section, "url": url}
                    data.extend(parent_child_chunker(re.sub(r'\[\w+\]', '', el.text.strip()), meta))
        time.sleep(1)
    return data

# --- 3. WIKIPEDIA SCRAPER ---
def scrape_wikipedia(urls):
    data = []
    for url in urls:
        print(f"Scraping Wikipedia: {url}")
        res = requests.get(url, headers=HEADERS)
        if res.status_code == 200:
            soup = BeautifulSoup(res.content, "html.parser")
            for tag in soup.find_all(['sup', 'span', 'div', 'table'], class_=re.compile(r'(reference|mw-editsection|noprint|thumb|navbox|infobox)')):
                tag.decompose()
                
            title = soup.find('h1', id='firstHeading').text.strip()
            content = soup.find('div', class_='mw-parser-output')
            if not content: continue
            
            section = "Summary"
            for el in content.find_all(['h2', 'h3', 'p']):
                if el.name in ['h2', 'h3']: section = el.text.strip()
                elif el.name == 'p' and len(el.text) > 15:
                    meta = {"source": "Wikipedia", "topic": title, "section": section, "url": url}
                    data.extend(parent_child_chunker(re.sub(r'\s+', ' ', el.text).strip(), meta))
        time.sleep(1)
    return data

if __name__ == "__main__":
    # Add your full URL lists here for the final run
    wiki_urls = [
        "https://en.wikipedia.org/wiki/Aloo_gobi",
        "https://en.wikipedia.org/wiki/Andhra_cuisine",
        "https://en.wikipedia.org/wiki/Anglo-Indian_cuisine",
        "https://en.wikipedia.org/wiki/Assamese_cuisine",
        "https://en.wikipedia.org/wiki/Awadhi_cuisine",
        "https://en.wikipedia.org/wiki/Balochi_cuisine",
        "https://en.wikipedia.org/wiki/Bangladeshi_cuisine",
        "https://en.wikipedia.org/wiki/Bengali_cuisine",
        "https://en.wikipedia.org/wiki/Bhojpuri_cuisine",
        "https://en.wikipedia.org/wiki/Bihari_cuisine",
        "https://en.wikipedia.org/wiki/Chettinad_cuisine",
        "https://en.wikipedia.org/wiki/Chutney",
        "https://en.wikipedia.org/wiki/Cuisine_of_Uttar_Pradesh",
        "https://en.wikipedia.org/wiki/Dal",
        "https://en.wikipedia.org/wiki/Dosa_(food)",
        "https://en.wikipedia.org/wiki/East_India#Cuisine",
        "https://en.wikipedia.org/wiki/Goan_Catholic_cuisine",
        "https://en.wikipedia.org/wiki/Goan_cuisine",
        "https://en.wikipedia.org/wiki/Gujarati_cuisine",
        "https://en.wikipedia.org/wiki/Hazara_cuisine",
        "https://en.wikipedia.org/wiki/Hyderabadi_cuisine",
        "https://en.wikipedia.org/wiki/Indian_Chinese_cuisine",
        "https://en.wikipedia.org/wiki/Indian_South_Africans#Cuisine",
        "https://en.wikipedia.org/wiki/Indian_cuisine",
        "https://en.wikipedia.org/wiki/Indian_fast_food",
        "https://en.wikipedia.org/wiki/Jain_vegetarianism",
        "https://en.wikipedia.org/wiki/Jharkhandi_cuisine",
        "https://en.wikipedia.org/wiki/Karahi",
        "https://en.wikipedia.org/wiki/Karnataka_cuisine",
        "https://en.wikipedia.org/wiki/Kashmiri_cuisine",
        "https://en.wikipedia.org/wiki/Kerala_cuisine",
        "https://en.wikipedia.org/wiki/Kodava_people",
        "https://en.wikipedia.org/wiki/Lahori_cuisine",
        "https://en.wikipedia.org/wiki/Maharashtrian_cuisine",
        "https://en.wikipedia.org/wiki/Maithil_cuisine",
        "https://en.wikipedia.org/wiki/Malaysian_Indian_cuisine",
        "https://en.wikipedia.org/wiki/Maldivian_cuisine",
        "https://en.wikipedia.org/wiki/Malvani_cuisine",
        "https://en.wikipedia.org/wiki/Mangalorean_Catholic_Cuisine",
        "https://en.wikipedia.org/wiki/Manipuri_cuisine",
        "https://en.wikipedia.org/wiki/Meitei_cuisine",
        "https://en.wikipedia.org/wiki/Mizo_cuisine",
        "https://en.wikipedia.org/wiki/Mughlai_cuisine",
        "https://en.wikipedia.org/wiki/Muhajir_cuisine",
        "https://en.wikipedia.org/wiki/Naan",
        "https://en.wikipedia.org/wiki/Naga_cuisine",
        "https://en.wikipedia.org/wiki/Nepalese_cuisine",
        "https://en.wikipedia.org/wiki/Newari_cuisine",
        "https://en.wikipedia.org/wiki/North_East_Indian_cuisine",
        "https://en.wikipedia.org/wiki/North_Indian_cuisine",
        "https://en.wikipedia.org/wiki/North_Karnataka",
        "https://en.wikipedia.org/wiki/Odia_cuisine",
        "https://en.wikipedia.org/wiki/Oriya_cuisine",
        "https://en.wikipedia.org/wiki/Pakistani_cuisine",
        "https://en.wikipedia.org/wiki/Parsi_cuisine",
        "https://en.wikipedia.org/wiki/Pashtun_cuisine",
        "https://en.wikipedia.org/wiki/Punjabi_cuisine",
        "https://en.wikipedia.org/wiki/Raita",
        "https://en.wikipedia.org/wiki/Rajasthani_cuisine",
        "https://en.wikipedia.org/wiki/Rice",
        "https://en.wikipedia.org/wiki/Sambar_(dish)",
        "https://en.wikipedia.org/wiki/Saraiki_cuisine",
        "https://en.wikipedia.org/wiki/Seekh_kebab",
        "https://en.wikipedia.org/wiki/Shahi_paneer",
        "https://en.wikipedia.org/wiki/Sikkimese_cuisine",
        "https://en.wikipedia.org/wiki/Sindhi_cuisine",
        "https://en.wikipedia.org/wiki/South_Asian_cuisine",
        "https://en.wikipedia.org/wiki/South_Indian_cuisine",
        "https://en.wikipedia.org/wiki/South_Karnataka",
        "https://en.wikipedia.org/wiki/Sri_Lankan_cuisine",
        "https://en.wikipedia.org/wiki/Tamil_cuisine",
        "https://en.wikipedia.org/wiki/Telugu_cuisine",
        "https://en.wikipedia.org/wiki/Thali",
        "https://en.wikipedia.org/wiki/Tripuri_cuisine",
        "https://en.wikipedia.org/wiki/Udupi_cuisine"
    ] 
    wikibooks_urls = [
    "https://en.wikibooks.org/wiki/Cookbook:Afghan_Bread",
    "https://en.wikibooks.org/wiki/Cookbook:Chicken_Tikka",
    "https://en.wikibooks.org/wiki/Cookbook:Naan",
    "https://en.wikibooks.org/wiki/Cookbook:Arisa_Pitha_(Fried_Indian_Sweet_Rice_Pastry)",
    "https://en.wikibooks.org/wiki/Cookbook:Chyapa_Shutki_Bharta",
    "https://en.wikibooks.org/wiki/Cookbook:Bhuna_Khichuri_(Bengali_Rice_and_Lentils)",
    "https://en.wikibooks.org/wiki/Cookbook:Mishti_Doi_(Bengali_Sweetened_Yogurt)",
    "https://en.wikibooks.org/wiki/Cookbook:Murghi_Korma_(Chicken_Korma)",
    "https://en.wikibooks.org/wiki/Cookbook:Pudina_Hilsa_(Bengali_Fish_with_Mint)",
    "https://en.wikibooks.org/wiki/Cookbook:Rosogulla_(Bengali_Milk_Balls_in_Syrup)",
    "https://en.wikibooks.org/wiki/Cookbook:Fried_Wheat_Bread_Balls_(Bhatoora)",
    "https://en.wikibooks.org/wiki/Cookbook:Makki_di_Roti_(Indian_Cornmeal_Flatbread)",
    "https://en.wikibooks.org/wiki/Cookbook:Potato_and_Cauliflower_Curry_(Aloo_Gobi)",
    "https://en.wikibooks.org/wiki/Cookbook:Salty_(Namkin)_Lassi",
    "https://en.wikibooks.org/wiki/Cookbook:Tandoori_Masala",
    "https://en.wikibooks.org/wiki/Cookbook:Appam_(Fermented_Rice_Pancake)",
    "https://en.wikibooks.org/wiki/Cookbook:Bonda_(South_Indian_Vegetable_Fritter)",
    "https://en.wikibooks.org/wiki/Cookbook:Hyderabad_Biryani",
    "https://en.wikibooks.org/wiki/Cookbook:Hyderabadi_Fried_Bread_with_Syrup_and_Nuts_(Double_ka_meetha)",
    "https://en.wikibooks.org/wiki/Cookbook:Idiyappam_(South_Indian_Rice_Noodles)",
    "https://en.wikibooks.org/wiki/Cookbook:Idli_(Steamed_Rice_and_Black_Gram_Bread)",
    "https://en.wikibooks.org/wiki/Cookbook:Kesari_(South_Indian_Semolina_Pudding)",
    "https://en.wikibooks.org/wiki/Cookbook:Khara_Pongal_(Rice_and_Mung_Bean_Porridge)",
    "https://en.wikibooks.org/wiki/Cookbook:Ragi_Dosa_(South_Indian_Millet_and_Rice_Pancake)",
    "https://en.wikibooks.org/wiki/Cookbook:Tamate_Ka_Kut_(Hyderabadi_Tomato_Curry)",
    "https://en.wikibooks.org/wiki/Cookbook:Chukauni_(Nepalese_Potato_Salad)",
    "https://en.wikibooks.org/wiki/Cookbook:Jhilinga_(Nepalese_Rice_Fritters)",
    "https://en.wikibooks.org/wiki/Cookbook:Masyaura_(Nepali_Fermented_Vegetable_Balls)",
    "https://en.wikibooks.org/wiki/Cookbook:Tibetan_Meat_Momos",
    "https://en.wikibooks.org/wiki/Cookbook:Aloo_Tikki_(Spiced_Potato_Patties)",
    "https://en.wikibooks.org/wiki/Cookbook:Basin_Ki_Kadi_(Sindhi_Chickpea_Flour_Curry)",
    "https://en.wikibooks.org/wiki/Cookbook:Chola_and_Roti",
    "https://en.wikibooks.org/wiki/Cookbook:Gobi_Bhagi_(Spiced_Cauliflower_Stew)",
    "https://en.wikibooks.org/wiki/Cookbook:Mustard_and_Curry_Leaf_Lassi",
    "https://en.wikibooks.org/wiki/Cookbook:Plain_Lassi",
    "https://en.wikibooks.org/wiki/Cookbook:Rajma_(Sindhi_Kidney_Bean_Curry)",
    "https://en.wikibooks.org/wiki/Cookbook:Sai_Bhaji_(Sindhi_Vegetable_Curry)",
    "https://en.wikibooks.org/wiki/Cookbook:Seviyan_Ji_Khirni_(Sindhi_Vermicelli_Pudding)",
    "https://en.wikibooks.org/wiki/Cookbook:Sindhi_Chickpea_Confection",
    "https://en.wikibooks.org/wiki/Cookbook:Sindhi_Fried_Potatoes",
    "https://en.wikibooks.org/wiki/Cookbook:Sindhi_Pulao",
    "https://en.wikibooks.org/wiki/Cookbook:Sindhi_Raitha",
    "https://en.wikibooks.org/wiki/Cookbook:Sindhi_Spiced_Fish",
    "https://en.wikibooks.org/wiki/Cookbook:Sindhi_Spiced_Moong_Dal",
    "https://en.wikibooks.org/wiki/Cookbook:Sindhi_Vegetable_Kofta",
    "https://en.wikibooks.org/wiki/Cookbook:Sylheti_Beef_Curry",
    "https://en.wikibooks.org/wiki/Cookbook:Sylheti_Biryani",
    "https://en.wikibooks.org/wiki/Cookbook:Sylheti_Doner_Kebab",
    "https://en.wikibooks.org/wiki/Cookbook:Sylheti_Fried_Rice",
    "https://en.wikibooks.org/wiki/Cookbook:Sylheti_Handesh",
    "https://en.wikibooks.org/wiki/Cookbook:Sylheti_Onion_and_Rice_Fritters",
    "https://en.wikibooks.org/wiki/Cookbook:Sylheti_Rice_Pudding",
    "https://en.wikibooks.org/wiki/Cookbook:Sylheti_Tangy_Curry",
    "https://en.wikibooks.org/wiki/Cookbook:Butter_Tea",
    "https://en.wikibooks.org/wiki/Cookbook:Corom_Chatni_(Mango_Chutney_with_Hot_Chillies)",
    "https://en.wikibooks.org/wiki/Cookbook:Dum_ka_Qimah_(Spiced_Minced_Meat)",
    "https://en.wikibooks.org/wiki/Cookbook:Khatti_Dal_(Spiced_Tamarind_Pigeon_Peas)",
    "https://en.wikibooks.org/wiki/Cookbook:Malpua_(South_Asian_Sweet_Pancake)",
    "https://en.wikibooks.org/wiki/Cookbook:Mango_Chutney_(Chunky)",
    "https://en.wikibooks.org/wiki/Cookbook:Mango_Chutney_(Smooth)",
    "https://en.wikibooks.org/wiki/Cookbook:Masala_Chai_II",
    "https://en.wikibooks.org/wiki/Cookbook:Mild_Salty_Lassi",
    "https://en.wikibooks.org/wiki/Cookbook:Papadam_(Black_Gram_Flatbread)",
    "https://en.wikibooks.org/wiki/Cookbook:Papaya_Lassi",
    "https://en.wikibooks.org/wiki/Cookbook:Papri_Chaat_(Crispy_Indian_Snack_with_Potato)",
    "https://en.wikibooks.org/wiki/Cookbook:Phulourie_(Split_Pea_Fritters)",
    "https://en.wikibooks.org/wiki/Cookbook:Prawn_Curry",
    "https://en.wikibooks.org/wiki/Cookbook:Qabuli_(Central_Asian_Rice_Pilaf)",
    "https://en.wikibooks.org/wiki/Cookbook:Sweet_Lassi",
    "https://en.wikibooks.org/wiki/Cookbook:Sweet_Mango_Lassi",
    "https://en.wikibooks.org/wiki/Cookbook:Watalappam_(Sri_Lankan_Coconut_Custard)"
]
    blog_urls = [
    "https://aroundtheworldin80cuisinesblog.wordpress.com/category/06-northern-india/",
    "https://aroundtheworldin80cuisinesblog.wordpress.com/category/38-afghanistan/",
    "https://aroundtheworldin80cuisinesblog.wordpress.com/category/44-southern-india/",
    "https://aroundtheworldin80cuisinesblog.wordpress.com/category/54-pakistan/",
    "https://aroundtheworldin80cuisinesblog.wordpress.com/category/20-sri-lanka-and-the-maldives/",
]
    
    print("--- Scraping Wikipedia ---")
    wiki_data = scrape_wikipedia(wiki_urls)
    with open("wikipedia_corpus.json", "w", encoding="utf-8") as f:
        json.dump(wiki_data, f, indent=4, ensure_ascii=False)
        
    print("\n--- Scraping Wikibooks ---")
    wikibooks_data = scrape_wikibooks(wikibooks_urls)
    with open("wikibooks_corpus.json", "w", encoding="utf-8") as f:
        json.dump(wikibooks_data, f, indent=4, ensure_ascii=False)
        
    print("\n--- Scraping Blog ---")
    blog_data = scrape_blog(blog_urls)
    with open("blog_corpus.json", "w", encoding="utf-8") as f:
        json.dump(blog_data, f, indent=4, ensure_ascii=False)
        
    print("\n✅ Ingestion Complete! Saved to 3 separate JSON files.")