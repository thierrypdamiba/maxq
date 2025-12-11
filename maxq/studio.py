import streamlit as st
import os
import pandas as pd
from dotenv import load_dotenv, find_dotenv
from maxq.search_engine import MaxQEngine, CollectionStrategy, SearchRequest

# Load environment variables
env_path = find_dotenv(usecwd=True)
if env_path:
    load_dotenv(env_path)
else:
    parent_env = os.path.abspath(os.path.join(os.getcwd(), "..", ".env"))
    if os.path.exists(parent_env):
        load_dotenv(parent_env)

st.set_page_config(page_title="MaxQ Studio", layout="wide", page_icon="⚡")

st.sidebar.title("⚡ MaxQ Studio")

@st.cache_resource
def get_engine():
    return MaxQEngine(
        qdrant_url=os.getenv("QDRANT_URL"),
        qdrant_api_key=os.getenv("QDRANT_API_KEY"),
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

try:
    engine = get_engine()
    collections_resp = engine.client.get_collections()
    collection_names = [c.name for c in collections_resp.collections]
    st.sidebar.success(f"Connected ({len(collection_names)} collections)")
except Exception as e:
    st.sidebar.error(f"Connection failed: {e}")
    collection_names = []
    engine = None

selected_collection = st.sidebar.selectbox("Select Collection", collection_names)

if selected_collection and engine:
    st.title(f"Collection: {selected_collection}")

    tab1, tab2, tab3 = st.tabs(["Overview", "Data", "Search"])

    with tab1:
        st.header("Overview")
        try:
            col_info = engine.client.get_collection(selected_collection)
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Points", col_info.points_count)
            with col2:
                st.metric("Status", col_info.status.name)
            with col3:
                st.metric("Vectors", str(col_info.config.params.vectors))
            with st.expander("Full Configuration"):
                st.json(col_info.model_dump())
        except Exception as e:
            st.error(f"Could not fetch collection info: {e}")

    with tab2:
        st.header("Data Preview")
        try:
            points, _ = engine.client.scroll(
                collection_name=selected_collection,
                limit=20,
                with_payload=True,
                with_vectors=False
            )
            if points:
                data = [p.payload for p in points]
                df = pd.DataFrame(data)
                st.dataframe(df)
            else:
                st.info("No data found.")
        except Exception as e:
            st.error(f"Could not fetch data: {e}")

    with tab3:
        st.header("Search")
        col1, col2 = st.columns([3, 1])
        with col1:
            query = st.text_input("Enter search query")
        with col2:
            limit = st.number_input("Limit", min_value=1, max_value=100, value=10)

        strategy = st.radio("Strategy", ["hybrid", "dense", "sparse"], horizontal=True)

        if st.button("Search", type="primary"):
            if query:
                config = CollectionStrategy(collection_name=selected_collection)
                req = SearchRequest(query=query, limit=limit, strategy=strategy)

                with st.spinner("Searching..."):
                    try:
                        results = engine.query(config, req)
                        if not results:
                            st.warning("No results found.")

                        for hit in results:
                            with st.container():
                                st.subheader(f"Score: {hit.score:.4f}")
                                text = hit.payload.get("_text", "")
                                if text:
                                    st.markdown(f"**Text:** {text[:500]}...")
                                with st.expander("Metadata"):
                                    st.json(hit.payload)
                                st.divider()
                    except Exception as e:
                        st.error(f"Search failed: {e}")
            else:
                st.warning("Please enter a query.")
else:
    st.info("Please select a collection from the sidebar.")
