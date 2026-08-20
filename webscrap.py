from langchain_community.document_loaders import WebBaseLoader
from bs4 import BeautifulSoup

# Define the URLs you want to load
urls = [
    "https://www.4mation.in",
    "https://www.4mation.in/index.html",
    "https://www.4mation.in/about.html",
    "https://www.4mation.in/network.html",
    "https://www.4mation.in/capabilities.html",
    "https://www.4mation.in/amenities.html",
    "https://www.4mation.in/infrastructure.html",
    "https://www.4mation.in/contact.html"
]

# Initialize the loader
loader = WebBaseLoader(urls)

# Load the web pages into LangChain Document objects
docs = loader.load()

locations = ["districts (Distribution Locations:\n- New Town · HQ\n- Kolkata\n- Howrah\n- Barasat\n- Sundarbans\n- Diamond Hbr.\n- Tamluk\n- Digha\n- Kharagpur\n- Jhargram\n- Bankura\n- Purulia\n- Durgapur\n- Asansol\n- Bardhaman\n- Suri\n- Krishnanagar\n- Berhampore\n- Malda\n- Raiganj\n- Balurghat\n- Siliguri\n- Cooch Behar\n- Alipurduar\n- Darjeeling\n- Kalimpong)"]
network_doc = next(
    doc for doc in docs
    if doc.metadata["source"] == "https://www.4mation.in/network.html"
)

network_doc.page_content += "\n\nDistricts in which distributions of network is present:\n"

network_doc.page_content += "\n".join(
    f"- {location}" for location in locations
)

#print(docs)