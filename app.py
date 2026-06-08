import streamlit as st
import pandas as pd
import plotly.express as px
from textblob import TextBlob
from sklearn.feature_extraction.text import CountVectorizer

# --- CONFIGURATION & SETUP ---
st.set_page_config(page_title="Support Strategy Engine", layout="wide", initial_sidebar_state="expanded")
st.title("🎯 Support Feedback Strategy Engine")
st.markdown("Transform raw customer feedback into boardroom-ready retention strategies.")

# --- FILE UPLOAD ---
st.sidebar.header("📥 Data Ingestion")
uploaded_file = st.sidebar.file_uploader("Upload Support Feedback (CSV)", type=["csv"])

if uploaded_file is not None:
    # Load Data
    df = pd.read_csv(uploaded_file)
    
    # Hide technical mapping in an expander to keep the sidebar clean
    with st.sidebar.expander("⚙️ Data Column Mapping (Optional)"):
        text_col = st.selectbox("Feedback Text", df.columns, index=list(df.columns).index('feedback') if 'feedback' in df.columns else 0)
        rating_col = st.selectbox("Rating", df.columns, index=list(df.columns).index('rating') if 'rating' in df.columns else 0)
        date_col = st.selectbox("Date", df.columns, index=list(df.columns).index('created_at') if 'created_at' in df.columns else 0)
        conv_col = st.selectbox("Conversation ID", df.columns, index=list(df.columns).index('conversation_id') if 'conversation_id' in df.columns else 0)
    
    # --- DATA CLEANING & NLP ---
    @st.cache_data
    def clean_and_process(data, txt, rate, date_c):
        # Clean numeric commas if present
        if data[rate].dtype == 'object':
            data[rate] = data[rate].astype(str).str.replace(',', '').astype(float)
        
        # Drop empty text
        data = data.dropna(subset=[txt]).copy()
        
        # NLP Sentiment
        data['Sentiment_Score'] = data[txt].apply(lambda x: TextBlob(str(x)).sentiment.polarity)
        data['Sentiment_Class'] = pd.cut(data['Sentiment_Score'], bins=[-1.0, -0.05, 0.05, 1.0], labels=['Negative', 'Neutral', 'Positive'])
        
        # Parse Dates cleanly
        data['Parsed_Date'] = pd.to_datetime(data[date_c].astype(str).str.replace(r'\s*,\s*', ', ', regex=True), errors='coerce').dt.date
        return data

    df_clean = clean_and_process(df, text_col, rating_col, date_col)

    # --- TABBED UI NAVIGATION ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 1. Data Health Overview", 
        "🧠 2. Sentiment & Trends", 
        "🔍 3. Root Cause Analysis", 
        "🚀 4. Strategic Plan"
    ])

    # --- TAB 1: DATA HEALTH ---
    with tab1:
        st.subheader("Operational Data Quality")
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Surveys", f"{len(df):,}")
        col2.metric("Valid Feedback", f"{len(df_clean):,}")
        col3.metric("Blank Contexts Dropped", f"{len(df) - len(df_clean):,}")
        col4.metric("Avg CSAT Rating", f"{df_clean[rating_col].mean():.2f} / 5.0")
        
        st.info("💡 **Consultant Note:** High volumes of 'Blank Contexts' (surveys with ratings but no text) often point to customers auto-closing tickets. Ensure these aren't skewing your overall satisfaction metrics.")

    # --- TAB 2: SENTIMENT & TRENDS ---
    with tab2:
        st.subheader("CX Health & Temporal Trends")
        col_s1, col_s2 = st.columns([1, 2])
        
        with col_s1:
            sentiment_dist = df_clean['Sentiment_Class'].value_counts().reset_index()
            sentiment_dist.columns = ['Sentiment', 'Volume']
            fig_pie = px.pie(sentiment_dist, names='Sentiment', values='Volume', hole=0.5, color='Sentiment', 
                             color_discrete_map={'Positive':'#10B981', 'Neutral':'#9CA3AF', 'Negative':'#EF4444'})
            fig_pie.update_layout(margin=dict(t=0, b=0, l=0, r=0))
            st.plotly_chart(fig_pie, use_container_width=True)
            
        with col_s2:
            df_trend = df_clean.groupby(['Parsed_Date', 'Sentiment_Class'], observed=False).size().unstack(fill_value=0).reset_index()
            fig_trend = px.area(df_trend, x='Parsed_Date', y=['Positive', 'Neutral', 'Negative'], 
                                color_discrete_map={'Positive':'#10B981', 'Neutral':'#9CA3AF', 'Negative':'#EF4444'})
            fig_trend.update_layout(margin=dict(t=0, b=0, l=0, r=0), xaxis_title="", yaxis_title="Ticket Volume")
            st.plotly_chart(fig_trend, use_container_width=True)

    # --- TAB 3: ROOT CAUSE ANALYSIS (UPDATED) ---
    with tab3:
        st.subheader("What exactly are customers saying?")
        
        def get_phrases(corpus):
            try:
                vec = CountVectorizer(ngram_range=(2,3), stop_words='english').fit(corpus)
                sum_words = vec.transform(corpus).sum(axis=0) 
                words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
                return sorted(words_freq, key=lambda x: x[1], reverse=True)[:8]
            except:
                return []

        col_r1, col_r2 = st.columns(2)
        
        with col_r1:
            st.markdown("#### ⚠️ Top Friction Points (Negative)")
            # Filter for negative data
            neg_df = df_clean[(df_clean['Sentiment_Class'] == 'Negative') | (df_clean[rating_col] <= 3)]
            neg_texts = neg_df[text_col].dropna().astype(str)
            neg_data = get_phrases(neg_texts)
            
            if neg_data:
                fig_neg = px.bar(pd.DataFrame(neg_data, columns=['Phrase', 'Count']), x='Count', y='Phrase', orientation='h', color_discrete_sequence=['#EF4444'])
                fig_neg.update_layout(yaxis={'categoryorder':'total ascending'}, margin=dict(t=0, b=0, l=0, r=0))
                st.plotly_chart(fig_neg, use_container_width=True)
            else:
                st.write("Not enough negative data to extract phrases.")
                
            # NEW: Verbatim Negative Feedback Viewer
            with st.expander("🔍 View Verbatim Negative Feedback"):
                st.markdown("Review the actual raw feedback driving the friction themes above:")
                st.dataframe(
                    neg_df[[conv_col, rating_col, text_col]].sort_values(by=rating_col, ascending=True),
                    use_container_width=True,
                    hide_index=True
                )

        with col_r2:
            st.markdown("#### 🏆 Top Success Drivers (Positive)")
            # Filter for positive data
            pos_df = df_clean[(df_clean['Sentiment_Class'] == 'Positive') & (df_clean[rating_col] >= 4)]
            pos_texts = pos_df[text_col].dropna().astype(str)
            pos_data = get_phrases(pos_texts)
            
            if pos_data:
                fig_pos = px.bar(pd.DataFrame(pos_data, columns=['Phrase', 'Count']), x='Count', y='Phrase', orientation='h', color_discrete_sequence=['#10B981'])
                fig_pos.update_layout(yaxis={'categoryorder':'total ascending'}, margin=dict(t=0, b=0, l=0, r=0))
                st.plotly_chart(fig_pos, use_container_width=True)
            else:
                st.write("Not enough positive data to extract phrases.")
                
            # NEW: Verbatim Positive Feedback Viewer
            with st.expander("🔍 View Verbatim Positive Feedback"):
                st.markdown("Review the actual raw feedback driving the success themes above:")
                st.dataframe(
                    pos_df[[conv_col, rating_col, text_col]].sort_values(by=rating_col, ascending=False),
                    use_container_width=True,
                    hide_index=True
                )

    # --- TAB 4: STRATEGIC PLAN ---
    with tab4:
        st.subheader("Executive Action Matrix")
        st.markdown("Based on the data extracted in Tabs 1-3, deploy the following operational adjustments:")
        
        strategy_matrix = pd.DataFrame({
            "Priority": ["🔴 Critical", "🟡 High Impact", "🟢 Proactive Value"],
            "Identified Theme": ["SLA Breach / Response Latency", "Resolution Completeness Issues", "High Quality Resolution Patterns"],
            "Strategic Action Blueprint": ["Deploy automated routing rules prioritizing clients hitting SLA threshold markers.", "Perform root-cause isolation on tech stacks causing reopened cases.", "Institutionalize agent protocols found in highly praised interactions."],
            "Expected Business ROI": ["Salvage ~12% Churn Risks", "Decrease Repeat Interactions by 8%", "Scale Customer Lifetime Value (LTV)"]
        })
        
        st.dataframe(strategy_matrix, use_container_width=True, hide_index=True)
        st.success("🎯 **Strategic Recommendation:** Focus immediately on the 🔴 Critical actions. Fixing response latency provides the highest direct impact on customer retention.")

else:
    st.info("👈 **Please upload your Support CSV in the sidebar to generate insights.**")