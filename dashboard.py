import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load Data
customer_df = pd.read_csv('Customers.csv')
product_df = pd.read_csv('Products.csv')
transaction_df = pd.read_csv('Transactions.csv')

# Merge Data
df = pd.merge(pd.merge(product_df, transaction_df, on='ProductID'), customer_df, on='CustomerID')
df.drop(columns=['Price_y'], inplace=True)

# Initialize session state for page navigation
if "page" not in st.session_state:
    st.session_state.page = "EDA Insights"

# Navigation Function
def go_to(page_name):
    st.session_state.page = page_name
    st.rerun()  # Instantly refresh to apply the change

# Page Selection Logic
if st.session_state.page == "EDA Insights":
    st.title("E-Commerce Business Insights Dashboard")

    # Business Insight 1: Top Performing Region
    st.subheader("Top Performing Region")
    region_quantity = df.groupby('Region')['Quantity'].sum()
    fig, ax = plt.subplots()
    region_quantity.plot(kind='barh', color='skyblue', ax=ax)
    ax.set_xlabel('Total Quantity')
    ax.set_ylabel('Region')
    st.pyplot(fig)

    # Business Insight 2: Most Productive Region
    st.subheader("Most Productive Region")
    region_sales = df.groupby('Region')['TotalValue'].sum()
    fig, ax = plt.subplots()
    region_sales.plot(kind='barh', color='orange', ax=ax)
    ax.set_xlabel('Total Value (USD)')
    ax.set_ylabel('Region')
    st.pyplot(fig)

    # Business Insight 3: Most Required Product_Category (Region-Wise)
    st.subheader("Most Required Product Category (Region-Wise)")
    category = {'Books': 1, 'Electronics': 2, 'Clothing': 3, 'Home Decor': 4}
    df['ctg'] = df['Category'].map(category)
    product_in_demand = df.groupby(['Region', 'ctg'])['Quantity'].sum()

    fig, ax = plt.subplots(figsize=(6, 4))
    product_in_demand.plot(kind='barh', color='skyblue', ax=ax)
    ax.set_xlabel('Products (Quantity)')
    ax.set_ylabel('Category (Region)')
    st.pyplot(fig)

    # Business Insight 4: Most Productive Product_Category
    st.subheader("Most Productive Product Category")
    category_sales = df.groupby('Category')['TotalValue'].sum()
    fig, ax = plt.subplots()
    category_sales.plot(kind='barh', color='orange', ax=ax)
    ax.set_xlabel('Total Value (USD)')
    ax.set_ylabel('Category')
    st.pyplot(fig)

    # Business Insight 5: Most Productive Product_Category (Region-Wise)
    st.subheader("Most Productive Product Category (Region-Wise)")
    region_category_sales = df.groupby(['Region', 'ctg'])['TotalValue'].sum()

    fig, ax = plt.subplots(figsize=(6, 4))
    region_category_sales.plot(kind='barh', color='orange', ax=ax)
    ax.set_xlabel('Total Value (USD)')
    ax.set_ylabel('Category (Region)')
    st.pyplot(fig)

    # Navigation Button
    if st.button("Go to Customer Segmentation →"):
        go_to("Customer Segmentation")

    if st.button("Go to Lookalike Model →"):
        go_to("Lookalike Model")
    

elif st.session_state.page == "Customer Segmentation":
    st.title("Customer Segmentation Analysis")

    # Prepare Data for Clustering
    cs_df = pd.merge(customer_df, transaction_df, on=['CustomerID'])
    region = {'South America': 1, 'Asia': 2, 'North America': 3, 'Europe': 4}
    cs_df['Reg'] = cs_df['Region'].map(region)
    X = cs_df[['Reg', 'Quantity', 'TotalValue']]

    # Compute SSE for different values of k
    sse = []
    K = range(2, 11)
    for k in K:
        kmeans = KMeans(n_clusters=k, random_state=42)
        kmeans.fit(X)
        sse.append(kmeans.inertia_)

    # Plot SSE (Elbow Method)
    st.subheader("Elbow Method for Optimal Clusters")
    fig, ax = plt.subplots()
    ax.plot(K, sse, marker='o', linestyle='--')
    ax.set_xlabel('Number of Clusters (k)')
    ax.set_ylabel('SSE')
    ax.set_title('Elbow Method')
    st.pyplot(fig)

    # Perform Clustering with k=6
    km = KMeans(n_clusters=6, random_state=42)
    cs_df['Cluster_6'] = km.fit_predict(X)

    # Plot Customer Segmentation Clustering
    st.subheader("Customer Segmentation (k=6)")
    fig, ax = plt.subplots()
    scatter = ax.scatter(cs_df['Quantity'], cs_df['TotalValue'], c=cs_df['Cluster_6'], cmap='viridis')
    ax.set_xlabel('Quantity')
    ax.set_ylabel('TotalValue')
    ax.set_title('Customer Segmentation Clustering')
    fig.colorbar(scatter, label='Cluster')
    st.pyplot(fig)

    # Identify High-Value Customers
    cluster_summary = cs_df.groupby('Cluster_6')[['Quantity', 'TotalValue']].mean()
    high_value_cluster = cluster_summary['TotalValue'].idxmax()
    high_value_customers = cs_df[cs_df['Cluster_6'] == high_value_cluster]

    # Display High-Value Customers
    st.subheader("High-Value Customers")
    st.write(high_value_customers[['CustomerID', 'Region', 'Quantity', 'TotalValue', 'Cluster_6']])

    # Navigation Button
    if st.button("← Go to EDA Insights"):
        go_to("EDA Insights")
    
    if st.button("Go to Lookalike Model →"):
        go_to("Lookalike Model")

elif st.session_state.page == "Lookalike Model":
    st.title("Lookalike Model")

    # Prepare Lookalike Data
    pt_df = pd.merge(product_df, transaction_df, on=['ProductID'])
    pt_df_grp = pt_df.groupby('CustomerID').agg({
        'ProductName': lambda x: ' '.join(x),
        'Category': lambda x: ' '.join(x)
    }).reset_index()

    pt_df_grp['tags'] = pt_df_grp['ProductName'] + ' ' + pt_df_grp['Category']
    pt_df_grp.drop(pt_df_grp[pt_df_grp['tags'].isnull()].index, inplace=True)
    pt_df_grp.drop_duplicates(inplace=True)

    original_cut_df = pt_df_grp.drop(columns=['ProductName','Category'])
    df = pd.merge(customer_df, original_cut_df, on=['CustomerID'])

    # Initialize a tfidf object
    tfidf = TfidfVectorizer(max_features=200)

    # Transform the data
    vectorized_data = tfidf.fit_transform(df['tags'].values)
    similarity = cosine_similarity(vectorized_data)

    lookalikes = []

    # Iterate over the first 20 customers
    for idx in range(20):
        customer_id = df.iloc[idx]['CustomerID']
        scores = similarity[idx]
        sorted_scores = sorted(enumerate(scores), key=lambda x: x[1], reverse=True)[1:4]
        
        # Extract customer IDs and similarity scores for the top 3 similar customers
        lookalike_ids = [df.iloc[i[0]]['CustomerID'] for i in sorted_scores]
        similarity_scores = [round(i[1], 4) for i in sorted_scores]
        
        # Append the result as a new row
        lookalikes.append([customer_id, lookalike_ids, similarity_scores])

    lookalike_df = pd.DataFrame(lookalikes, columns=['CustomerID', 'LookalikeID', 'SimilarityScore'])

    # Display Lookalike Data
    st.subheader("Lookalike Customers")
    st.write(lookalike_df)

    # Navigation Button
    if st.button("← Go to Customer Segmentation"):
        go_to("Customer Segmentation")
    
    if st.button("← Go to EDA Insights"):
        go_to("EDA Insights")

