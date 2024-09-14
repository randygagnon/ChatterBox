import os
import json
import aiohttp
import asyncio
import nest_asyncio  # To allow nested asyncio loops in Streamlit
import streamlit as st
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import JSONLoader
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_experimental.text_splitter import SemanticChunker
from langchain.schema import Document

# Apply nest_asyncio to allow running asyncio in Streamlit
nest_asyncio.apply()

# Instantiate OpenAI client
load_dotenv()
openai_api_key = os.getenv('OPENAI_API_KEY')
llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")

class NetboxGiftwrap:
    """
    Class to interact with NetBox API and gather data.
    """

    def __init__(self, url, token):
        self.url = url
        self.token = token

    async def netbox_giftwrap(self):
        """
        Main method to gather data from NetBox APIs.
        """
        results = await self.main()
        return results

    def netbox_api_list(self):
        """
        Returns a list of NetBox API endpoints to fetch data from.
        """
        self.api_list = [
            "/api/ipam/aggregates/",
            "/api/ipam/asns/",
            "/api/dcim/cables/",
            "/api/circuits/circuit-terminations/",
            "/api/circuits/circuit-types/",
            "/api/circuits/circuits/",
            "/api/virtualization/cluster-groups/",
            "/api/virtualization/cluster-types/",
            "/api/virtualization/clusters/",
            "/api/dcim/console-port-templates/",
            "/api/dcim/console-ports/",
            "/api/tenancy/contact-assignments/",
            "/api/tenancy/contact-groups/",
            "/api/tenancy/contact-roles/",
            "/api/tenancy/contacts/",
            "/api/dcim/device-bay-templates/",
            "/api/dcim/device-bays/",
            "/api/dcim/device-roles/",
            "/api/dcim/device-types/",
            "/api/dcim/devices/",
            "/api/dcim/front-port-templates/",
            "/api/dcim/front-ports/",
            "/api/users/groups/",
            "/api/dcim/interface-templates/",
            "/api/dcim/interfaces/",
            "/api/dcim/inventory-items/",
            "/api/ipam/ip-addresses/",
            "/api/ipam/ip-ranges/",
            "/api/dcim/locations/",
            "/api/dcim/manufacturers/",
            "/api/dcim/module-bay-templates/",
            "/api/dcim/module-bays/",
            "/api/dcim/module-types/",
            "/api/dcim/modules/",
            "/api/dcim/platforms/",
            "/api/dcim/power-feeds/",
            "/api/dcim/power-outlet-templates/",
            "/api/dcim/power-outlets/",
            "/api/dcim/power-panels/",
            "/api/dcim/power-port-templates/",
            "/api/dcim/power-ports/",
            "/api/ipam/prefixes/",
            "/api/circuits/provider-networks/",
            "/api/circuits/providers/",
            "/api/dcim/rack-reservations/",
            "/api/dcim/rack-roles/",
            "/api/dcim/racks/",
            "/api/dcim/rear-port-templates/",
            "/api/dcim/rear-ports/",
            "/api/dcim/regions/",
            "/api/ipam/rirs/",
            "/api/ipam/roles/",
            "/api/ipam/route-targets/",
            "/api/ipam/service-templates/",
            "/api/ipam/services/",
            "/api/dcim/site-groups/",
            "/api/dcim/sites/",
            "/api/status/",
            "/api/tenancy/tenant-groups/",
            "/api/tenancy/tenants/",
            "/api/users/tokens/",
            "/api/users/users/",
            "/api/dcim/virtual-chassis/",
            "/api/virtualization/interfaces/",
            "/api/virtualization/virtual-machines/",
            "/api/ipam/vlan-groups/",
            "/api/ipam/vlans/",
            "/api/ipam/vrfs/"
        ]
        return self.api_list

    async def get_api(self, api_url):
        """
        Asynchronously fetch data from a single NetBox API endpoint.
        """
        payload = {}
        headers = {
            'Accept': 'application/json',
            'Authorization': f'Token { self.token }',
        }
        async with aiohttp.ClientSession() as session:
            try:
                async with session.get(f"{self.url}{api_url}", headers=headers, data=payload, ssl=False) as resp:
                    responseJSON = await resp.json()
                    if api_url == "/api/status/":
                        responseList = responseJSON
                    else:
                        responseList = responseJSON['results']
                        offset = 50
                        total_pages = responseJSON['count'] / 50
                        while total_pages > 1:
                            async with session.get(f"{self.url}{api_url}?limit=50&offset={offset}", headers=headers, data=payload, ssl=False) as resp:
                                responseDict = await resp.json()
                                responseList.extend(responseDict['results'])
                                offset += 50
                                total_pages -= 1
                return (api_url, responseList)
            except Exception as e:
                # Error handling
                st.error(f"Error fetching data from {api_url}: {e}")
                return (api_url, [])

    async def main(self):
        """
        Gather data from all specified NetBox API endpoints.
        """
        api_list = self.netbox_api_list()
        results = await asyncio.gather(*(self.get_api(api_url) for api_url in api_list))
        return results

class Chatterbox:
    """
    Class to handle conversation with the user based on NetBox data.
    """

    def __init__(self, data):
        self.conversation_history = []
        self.load_text(data)
        self.split_into_chunks()
        self.store_in_chroma()
        self.setup_conversation_memory()
        self.setup_conversation_retrieval_chain()

    def load_text(self, data):
        """
        Load NetBox data into documents for processing.
        """
        self.pages = []
        for item in data:
            api, payload = item
            text = json.dumps(payload)
            doc = Document(page_content=text, metadata={"source": api})
            self.pages.append(doc)

    def split_into_chunks(self):
        """
        Split documents into chunks using SemanticChunker.
        """
        self.text_splitter = SemanticChunker(OpenAIEmbeddings())
        self.docs = self.text_splitter.split_documents(self.pages)

    def store_in_chroma(self):
        """
        Store document embeddings in Chroma vector store.
        """
        embeddings = OpenAIEmbeddings()
        self.vectordb = Chroma.from_documents(self.docs, embedding=embeddings)
        self.vectordb.persist()

    def setup_conversation_memory(self):
        """
        Set up conversation memory to keep track of chat history.
        """
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    def setup_conversation_retrieval_chain(self):
        """
        Set up the conversational retrieval chain using LangChain.
        """
        self.qa = ConversationalRetrievalChain.from_llm(
            llm,
            self.vectordb.as_retriever(search_kwargs={"k": 25})
        )

    def chat(self, question):
        """
        Process a user question and return an answer.
        """
        # Generate a response using the ConversationalRetrievalChain
        try:
            response = self.qa({"question": question, "chat_history": self.conversation_history})
            answer = response.get('answer', 'No answer found.')
        except Exception as e:
            st.error(f"An error occurred during processing: {e}")
            answer = 'No answer found due to an error.'

        # Update the conversation history
        self.conversation_history.append((question, answer))

        # Update the Streamlit session state
        st.session_state['conversation_history'] += f"\nUser: {question}\nNetBox: {answer}"

        # Return the AI's response for immediate display
        return answer

    def format_conversation_history(self, include_current=True):
        """
        Format the conversation history for context.
        """
        formatted_history = ""
        history_to_format = self.conversation_history[:-1] if not include_current else self.conversation_history
        for msg in history_to_format:
            speaker = "You: " if msg["sender"] == "user" else "Bot: "
            formatted_history += f"{speaker}{msg['text']}\n"
        return formatted_history

# Load environment variables securely
netbox_url = os.getenv('NETBOX_URL')
netbox_token = os.getenv('NETBOX_TOKEN')

# Ensure sensitive information is not exposed
if not netbox_url or not netbox_token:
    st.error("NetBox URL and Token must be set in environment variables.")
    st.stop()

# Implement caching for expensive computations
@st.cache_data(show_spinner=False)
def fetch_netbox_data():
    """
    Fetch and cache NetBox data.
    """
    netbox = NetboxGiftwrap(url=netbox_url, token=netbox_token)
    data = asyncio.run(netbox.netbox_giftwrap())
    return data

@st.cache_resource
def get_chatterbox_instance(data):
    """
    Initialize and cache Chatterbox instance.
    """
    return Chatterbox(data)

# Page functions
def netbox_data_gathering_page():
    st.title("Gathering JSON data from NetBox")

    if 'data_gathered' not in st.session_state:
        st.session_state['data_gathered'] = False

    if not st.session_state['data_gathered']:
        with st.spinner('Gathering JSON data from NetBox...'):
            st.text("Please wait, this may take a moment...")
            try:
                data = fetch_netbox_data()
                st.session_state['data_gathered'] = True
                st.session_state['netbox_data'] = data
                st.success("Data successfully gathered! Proceed to the Q&A interface.")
            except Exception as e:
                st.error(f"An error occurred: {e}")
                st.stop()

    if st.button('Proceed to Chat with NetBox'):
        st.session_state['page'] = 'Q&A'
        # Use st.query_params.from_dict() to update query params
        st.query_params.from_dict({"page": st.session_state['page']})

def qa_page():
    st.title("ChatterBox - Talk to Your NetBox")

    if 'conversation_history' not in st.session_state:
        st.session_state['conversation_history'] = ""

    if 'netbox_data' not in st.session_state:
        st.error("NetBox data not found. Please run the data gathering step first.")
        return

    data = st.session_state['netbox_data']
    chat_instance = get_chatterbox_instance(data)

    user_input = st.text_input("Ask a question about NetBox:")
    if st.button("Ask"):
        with st.spinner('Processing...'):
            ai_response = chat_instance.chat(user_input)
            st.session_state['conversation_history'] += f"\nUser: {user_input}\nAI: {ai_response}"
            st.text_area("Conversation History:", value=st.session_state['conversation_history'], height=300)

# Main app flow
if 'page' not in st.session_state:
    st.session_state['page'] = 'NetBox Data Gathering'

# Display the correct page based on the session state
if st.session_state['page'] == 'NetBox Data Gathering':
    netbox_data_gathering_page()
elif st.session_state['page'] == 'Q&A':
    qa_page()
