import streamlit as st
import pandas as pd
from google import genai
from google.genai import types
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import constants


st.set_page_config(page_title="AI analyst", page_icon="📑")


API_KEY = st.secrets["google_api_key"]
client = genai.Client(api_key=API_KEY)

def generate_answer(client, prompt):
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=prompt,
        config=types.GenerateContentConfig( 
            temperature=0.3,
            max_output_tokens=1000
        )
    )
    return response.text

st.title("AI Data Analyst - Ask Anything!")
#--------------
# Dataset-info
#--------------

file = st.file_uploader("Upload CSV", type=["csv"])

if file:
    df= pd.read_csv(file)    
    st.success("File loaded successfully!")
    st.subheader("Data Preview")
    st.dataframe(df.sample(5))
    st.session_state["df"] = df
if "df" not in st.session_state:
    st.write("Please upload a file")
    st.stop()
   
    
col1, col2, col3, col4 = st.columns(4)
with col1:
        st.metric("Rows", len(df))
with col2:
        st.metric("Columns", len(df.columns))
with col3:
        st.metric("Missing Values", df.isnull().sum().sum())
with col4:
        st.metric("Unique Values", df.nunique().sum())
     

dup_count=df.duplicated().sum()
st.write(f"### Duplcated rows:{dup_count}")
if dup_count>0:
    st.info("Duplicates detected. You may want to remove them in cleaning phase.")


constant_col=[col for col in df.columns if df[col].nunique==1]
if constant_col:
    st.write("### Constant columns:",constant_col)
    st.info("These columns cnatin only one unique value.")
else:
    st.write("### Constant columns:")
    st.write("No constant columns found.")

    
st.divider()
st.write("### Data Types")
dtype_df = df.dtypes.reset_index()
dtype_df.columns = ["column", "dtype"]
dtype_df["dtype"] = dtype_df["dtype"].astype(str)   
st.dataframe(dtype_df)

st.divider()
num_col=df.select_dtypes(include=["int64","float64"]).columns.tolist()
cat_col=df.select_dtypes(include=["object"]).columns.tolist()
st.write("### Numeric columns:",num_col)
st.write("### Categorical columns:",cat_col)

st.divider()
st.subheader("Missing Value Percentage")
missing_pct = (df.isnull().mean() * 100).round(2)
missing_df = pd.DataFrame({
"column": missing_pct.index,
"missing_percentage": missing_pct.values
})
st.dataframe(missing_df.sort_values("missing_percentage", ascending=False))



    
st.divider()
st.subheader("📊 Numerical Summary Statistics")
st.dataframe(df.describe())
    
st.divider()
st.divider()
df=st.session_state["df"]

if "df_new" not in st.session_state:
    st.session_state["df_new"]=st.session_state["df"]
df_new=st.session_state["df_new"]

if "target_type" not in st.session_state:
    st.session_state.target_type = None
if "action_msg" not in st.session_state:
    st.session_state.action_msg = None


#--------------
# Cleaner-phase
#--------------
st.write("## Select where you want to perform operations")
op1,op2,op3=st.columns(3)
with op1:
    if st.button("📊 Columns", width='stretch'):
        st.session_state.target_type = "column"

with op2:
    if st.button("📄 Rows", width='stretch'):
        st.session_state.target_type = "row"
with op3:
    if st.button("📈 Outliers",width='stretch'):
        st.session_state.target_type="outliers"

if st.session_state.target_type == "column":
    st.subheader("🔽 Column Operations")

    target_col = st.selectbox(
        "Select column",
        df_new.columns
     )

    b1, b2, b3, b4 = st.columns(4)

    with b1:
        if st.button("🗑 Drop Column", use_container_width=True):

           if target_col in df_new.columns:
             st.session_state["df_new"] = df_new.drop(columns=[target_col])
             st.session_state.action_msg = ("success", f"Column '{target_col}' removed")

           else:
              st.session_state.action_msg = ("warning", f"Column '{target_col}' is already removed")

    with b2:
        if st.button("🔢 Mean",use_container_width=True):
          if target_col not in df_new.columns:
            st.session_state.action_msg=("warning",f"column {target_col} does not exist")
          elif df_new[target_col].dtype=='object':
            st.session_state.action_msg=("warning",f"column {target_col} is of object type")
          else:
            df_new[target_col]=df_new[target_col].fillna(df_new[target_col].mean())
            st.session_state["df_new"]=df_new
            st.session_state.action_msg=("success",f"Missing values are filled with mean in {target_col}")
         

        
    with b3:
       if st.button(" 📎Median",use_container_width=True):
        if target_col not in df_new.columns:
            st.session_state.action_msg=("warning",f"column {target_col} does not exist")
        elif df_new[target_col].dtype=='object':
           st.session_state.action_msg=("warning",f"column {target_col} is of object type")
        else:
            df_new[target_col]=df_new[target_col].fillna(df_new[target_col].median())
            st.session_state["df_new"]=df_new
            st.session_state.action_msg=("success",f"Missing values are filled with median in '{target_col}'")


    with b4:
        if st.button("🔢 Mode",use_container_width=True):
          if target_col not in df_new.columns:
            st.session_state.action_msg=("warning",f"column {target_col} does not exist")
          else:
            df_new[target_col]=df_new[target_col].fillna(df_new[target_col].mode())
            st.session_state["df_new"]=df_new
            st.session_state.action_msg=("success",f"Missing values are filled with mode in '{target_col}'")
    if st.session_state.action_msg:
      msg_type,msg= st.session_state.action_msg
      if msg_type=="success":
        st.success(msg)
      elif msg_type=="warning":
        st.warning(msg) 
   
elif st.session_state.target_type == "row":
    st.subheader("🔽 Row Operations")
    
    if st.button("Drop duplicate rows",use_container_width=True):
        dup=df_new.duplicated().sum()
        if dup>0:
             st.session_state.df_new=df_new.drop_duplicates()
             st.session_state.action_msg=("success",f"duplicated {dup} rows are removed")
        else:
             st.session_state.action_msg=("warning","rows are already removed")
    if st.button("remove rows with missing values ",use_container_width=True):
        mis_row=df_new.isnull().any(axis=1)
        count=mis_row.sum()
        

        if count>0:
             st.subheader("Preview rows to be removed")
             st.dataframe(df_new[mis_row])
             st.session_state.df_new=df_new.dropna(how="any")             
             st.session_state.action_msg=("success",f"{count} rows having missing values are removed")
        else:
             st.session_state.action_msg=("warning"," No rows with missing values is found")



    if st.session_state.action_msg:
     msg_type,msg= st.session_state.action_msg
     if msg_type=="success":
        st.success(msg)
     elif msg_type=="warning":
        st.warning(msg)
elif st.session_state.target_type == "outliers":

     st.subheader("🚨 Outlier Detection (IQR Method)")

     numeric_cols = df_new.select_dtypes(include=["int64", "float64"]).columns

     if len(numeric_cols) == 0:
        st.warning("No numeric columns available for outlier detection")
        st.stop()

     target_col = st.selectbox("Select numeric column", numeric_cols)

     q1 = df_new[target_col].quantile(0.25)
     q3 = df_new[target_col].quantile(0.75)
     iqr = q3 - q1

     lower_bound = q1 - 1.5 * iqr
     upper_bound = q3 + 1.5 * iqr

     outlier_mask = (df_new[target_col] < lower_bound) | (df_new[target_col] > upper_bound)
     outliers_df = df_new[outlier_mask]

     count = outliers_df.shape[0]
     percent = round((count / len(df_new)) * 100, 2)

     st.info(f"Detected **{count} outliers** ({percent}%) using IQR method")

     if count > 0:
        st.subheader("🔍 Outlier Preview")
        st.dataframe(outliers_df)

        b1, b2, b3 = st.columns(3)

        with b1:
            if st.button("🗑 Remove outliers", use_container_width=True):
                st.session_state.df_new = df_new[~outlier_mask]
                st.session_state.action_msg = (
                    "success",
                    f"{count} outliers removed from '{target_col}'"
                )

        with b2:
            if st.button("📌 Cap outliers", use_container_width=True):
                capped = df_new[target_col].clip(lower_bound, upper_bound)
                df_new[target_col] = capped
                st.session_state.df_new = df_new
                st.session_state.action_msg = (
                    "success",
                    f"Outliers capped using IQR bounds in '{target_col}'"
                )

        with b3:
            st.button("🚫 Ignore", use_container_width=True)
     else:
        st.success("No outliers detected 🎉")

   


st.header("Live Dataset preview")
st.dataframe(st.session_state.df_new,use_container_width=True)


st.markdown("""
    <style>
    button {
    border-radius: 12px !important;
    height: 3em;
    font-size: 16px;
    }
    </style>
    """, unsafe_allow_html=True)
                 

#--------------
# Visuals
#--------------   
st.divider()
st.divider()
st.subheader("Select chart to get visuals.")


st.write("## X-axis")
x=st.selectbox(
   "Select X-axis",df_new.columns
)
st.divider()
st.write("## Y-axis")
y=st.selectbox(
    "Select Y-axis",df_new.columns
)   
st.write("## Graph")
gp=['','Line','Bar-graph','Boxplot','CountPlot','Pie-chart','Histogram','Distplot','Scatterplot','Heatmap','Clustermap']

target_gp=st.selectbox(
     "Select Graph",
     gp
)

if target_gp:
    
    fig,ax=plt.subplots()
    if target_gp=='Bar-graph':
        unique_count=df_new[x].nunique()
        if unique_count>20:
         st.warning(f"{x} has uniques values more than 20. Chart may get cluttered,thus we take only 6 values to get idea ")
         sns.barplot(data=df_new,
          x=df_new[x].sample(6),
         y=df_new[y].sample(6),
         ax=ax)
        else:
            sns.barplot(data=df_new,x=x,y=y,ax=ax)
        ax.set_title(f"Bar-graph of {x} and {y}")
    elif target_gp=='Boxplot':
      unique_count=df_new[x].nunique()
      if unique_count>20:
        st.warning(f"{x} has uniques values more than 20. Chart may get cluttered,thus we take only 6 values to get idea")
        sns.boxplot(data=df_new,
        x=df_new[x].sample(6),
        y=df_new[y].sample(6),
        ax=ax)
      else:
       sns.boxplot(data=df_new,
       x=x,
       y=y,
       ax=ax)
      ax.set_title(f"Boxplot of {x} and {y}")
    elif target_gp=='CountPlot':
        unique_count=df_new[x].nunique()
        if unique_count>20:
         st.warning(f"{x} has uniques values more than 20. Chart may get cluttered,thus we take only 6 values to get idea ")
         sns.countplot(data=df_new,
         x=df_new[x].sample(6),
         y=df_new[y].sample(6),
         ax=ax)
        else:
         sns.countplot(data=df_new,x=x,ax=ax)
         ax.set_title(f"CountPlot of {x} and {y}")
    elif target_gp=='Histogram':
      plt.hist(df_new[x])
      ax.set_title(f"Histogram of {x}")
    elif target_gp=='Distplot':
      unique_count=df_new[x].nunique()
      if unique_count>20:
        st.warning(f"{x} has uniques values more than 20. Chart may get cluttered,thus we take only 6 values to get idea ")
        sns.distplot(data=df_new,
        x=df_new[x].sample(6),
        ax=ax)
      else:
       sns.distplot(data=df_new,
       x=x,
       ax=ax
      )
      ax.set_title(f"Distplot of {x}")
    elif target_gp=='Pie-chart':
      values=df_new[y].value_counts()
      if len(values) > 15:
        st.warning(f"{y} has high-cardinality")
      else:
       ax.pie(values.values,
       labels=values.index,
       autopct='%1.1f',
      )
       ax.set_title(f"Pie chart of {y}")
    elif target_gp == 'Scatterplot':
      temp = df_new.copy()

     
      temp[x] = pd.to_numeric(
        temp[x].astype(str).str.replace(r"[^\d\.-]", "", regex=True),
        errors='coerce'
     )
      temp[y] = pd.to_numeric(
        temp[y].astype(str).str.replace(r"[^\d\.-]", "", regex=True),
        errors='coerce'
     )

      temp = temp.dropna(subset=[x, y])

      st.write("Valid rows for scatter:", temp.shape[0])

      if temp.shape[0] == 0:
        st.warning("Selected X and Y cannot form a scatter plot")
      else:
        sns.scatterplot(data=temp, x=x, y=y, ax=ax)
        ax.set_title(f"Scatter Plot: {x} vs {y}")

    elif target_gp=='Heatmap':
      corr=df_new.select_dtypes(include='number').corr()
      sns.heatmap(corr, annot=True,cmap="coolwarm",
      ax=ax
      )
      ax.set_title(f"Heatmap of data")
    elif target_gp=='Clustermap':
      cluster_fig=sns.clustermap(
      df_new.select_dtypes(include='number'))
      st.pyplot(cluster_fig.fig)
    elif target_gp=='Line':
        if not pd.api.types.is_numeric_dtype(df_new[x]):
            st.warning("X axis must be numeric")
        else:
            temp=df_new.sort_values(by=x)
            sns.lineplot(data=temp,
            x=x,
            y=y,
            ax=ax,color="green")
            ax.set_title(f"Line chart of {x} and {y}")
    st.pyplot(fig)

#--------------
# Insights
#--------------
st.divider()
st.header(" Ask Anything About Your Data!")


if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("Ask about trends, predictions, comparisons, or anything!"):
 
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            context = ""
            if df is not None:
                context = f"""
   **Data Context:**
   - Shape: {df.shape}
   - Columns: {', '.join(df.columns.tolist())}
   - Sample data: {df.head(20).to_dict('records')}
   - Basic stats: {df.describe().to_dict()}
   """
            
            full_prompt = f"""
    You are an expert data analyst. Answer this question based on the data provided:

   **User Question:** {prompt}

   {context}

   Provide:
   1. Clear, actionable insights
   2. Specific numbers/references from data when possible
   3. Practical recommendations
   4. Bullet points for readability

   Keep it concise but thorough.
   """

            try:
                answer = generate_answer(client, full_prompt)
                st.markdown(answer)
                st.session_state.messages.append({"role": "assistant", "content": answer})
            except Exception as e:
                st.error(f"Error: {str(e)}")


if st.button("Clear Chat", use_container_width=True):
    st.session_state.messages = []
    st.rerun()

# #--------------
# # CSS
# #--------------
# st.markdown("""
# <style>
# /* App background */
# .stApp{
# background:linear-gradient(135deg,#0f172a,#020617);
# color:#e5e7b;
# }
# /*buttons*/
# .stButton>button{
# background:linear-gradient(135deg,#2563eb,#38bdf8);
# color:white;
# border:none;
# transition:all 0.25s ease-in-out;
# }
# /*buttons hover*/
# .stButton>button:hover{
# background:linear-gradient(135deg,#1e40af,#0ea5e9);
# transforn:scale(1.03);
# }
# /*Warning box*/
# .stAlert-warning{
# background-color:#3b1d0a;
# color:#fde68a;
# border-radius:10px;
# }


# </style>
# """,unsafe_allow_html=True)
 





