import streamlit as st
from langchain.document_loaders import PyPDFLoader, TextLoader
import fitz  # PyMuPDF
import pandas as pd
import numpy as np
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.chat_models import ChatOpenAI
from langchain.prompts import PromptTemplate
import os
import tempfile
import re

# Set page configuration
st.set_page_config(
    page_title="Academic & Career Advisor for Pakistani Students",
    page_icon="🎓",
    layout="wide"
)

# Define custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1E3A8A;
        text-align: center;
        margin-bottom: 1.5rem;
    }
    .sub-header {
        font-size: 1.8rem;
        color: #1E3A8A;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .section {
        background-color: #F3F4F6;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1.5rem;
    }
    .info-box {
        background-color: #E0F2FE;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 5px solid #0369A1;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<h1 class="main-header">Academic & Career Advisor for Pakistani Students</h1>', unsafe_allow_html=True)

st.markdown('<div class="info-box">This application analyzes student transcripts to provide personalized recommendations for degree programs, scholarships, and career paths based on academic performance and interests.</div>', unsafe_allow_html=True)

# Initialize session state variables
if 'extracted_data' not in st.session_state:
    st.session_state.extracted_data = None
if 'analysis_results' not in st.session_state:
    st.session_state.analysis_results = None
if 'api_key_set' not in st.session_state:
    st.session_state.api_key_set = False

# Sidebar for API key configuration
with st.sidebar:
    st.markdown("## Configuration")
    
    llm_option = st.selectbox(
        "Select LLM Provider",
        ["OpenAI (GPT-3.5)", "OpenAI (GPT-4)", "Anthropic Claude", "Grok"]
    )
    
    api_key = st.text_input("Enter API Key", type="password")
    
    if st.button("Save API Key"):
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
            st.session_state.api_key_set = True
            st.success("API Key saved successfully!")
        else:
            st.error("Please enter a valid API key")

    st.markdown("---")
    st.markdown("### About")
    st.markdown("""
    This application helps Pakistani students analyze their academic records to:
    - Find suitable degree programs
    - Discover scholarship opportunities
    - Explore career paths
    - Get personalized recommendations
    """)

# Function to extract text from PDF
def extract_text_from_pdf(pdf_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
        tmp_file.write(pdf_file.getvalue())
        tmp_file_path = tmp_file.name
    
    doc = fitz.open(tmp_file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    
    os.unlink(tmp_file_path)
    return text

# Function to extract transcript data
def extract_transcript_data(text):
    # Extract student information
    name_match = re.search(r"Name:?\s*([A-Za-z\s]+)", text)
    student_name = name_match.group(1).strip() if name_match else "Not found"
    
    # Extract subjects and grades
    # This is a simplified pattern, would need to be adapted to actual transcript format
    subject_pattern = r"([A-Za-z\s]+)\s+([A-F][+-]?|[0-9]+(?:\.[0-9]+)?)"
    subjects_grades = re.findall(subject_pattern, text)
    
    # Extract GPA/CGPA
    gpa_match = re.search(r"GPA:?\s*([0-9]+\.[0-9]+)", text)
    cgpa_match = re.search(r"CGPA:?\s*([0-9]+\.[0-9]+)", text)
    
    gpa = gpa_match.group(1) if gpa_match else "Not found"
    cgpa = cgpa_match.group(1) if cgpa_match else gpa if gpa != "Not found" else "Not found"
    
    # Prepare data structure
    data = {
        "student_name": student_name,
        "subjects": [subject for subject, _ in subjects_grades],
        "grades": [grade for _, grade in subjects_grades],
        "gpa": gpa,
        "cgpa": cgpa,
        "full_text": text
    }
    
    return data

# Function to get subject strengths
def analyze_subject_strengths(data):
    subjects = data['subjects']
    grades = data['grades']
    
    # Convert letter grades to numerical values if needed
    grade_mapping = {
        'A+': 4.0, 'A': 4.0, 'A-': 3.7,
        'B+': 3.3, 'B': 3.0, 'B-': 2.7,
        'C+': 2.3, 'C': 2.0, 'C-': 1.7,
        'D+': 1.3, 'D': 1.0, 'D-': 0.7,
        'F': 0.0
    }
    
    # Try to convert grades to numerical values
    numerical_grades = []
    for grade in grades:
        try:
            # If it's already a number
            numerical_grades.append(float(grade))
        except ValueError:
            # If it's a letter grade
            if grade in grade_mapping:
                numerical_grades.append(grade_mapping[grade])
            else:
                # Fallback value
                numerical_grades.append(2.0)
    
    # Create a DataFrame
    df = pd.DataFrame({
        'subject': subjects,
        'grade': numerical_grades
    })
    
    # Sort by grade
    df = df.sort_values(by='grade', ascending=False)
    
    subject_categories = {
        'STEM': ['Mathematics', 'Physics', 'Chemistry', 'Biology', 'Computer Science', 'Statistics'],
        'Humanities': ['English', 'History', 'Geography', 'Literature', 'Philosophy'],
        'Social Sciences': ['Economics', 'Psychology', 'Sociology', 'Political Science'],
        'Business': ['Accounting', 'Business Studies', 'Commerce', 'Finance'],
        'Arts': ['Art', 'Music', 'Drama', 'Design']
    }
    
    # Categorize subjects
    df['category'] = 'Other'
    for category, subject_list in subject_categories.items():
        for subject in subject_list:
            df.loc[df['subject'].str.contains(subject, case=False), 'category'] = category
    
    # Find strength areas
    strength_categories = df.groupby('category')['grade'].mean().sort_values(ascending=False)
    
    return {
        'subject_performance': df.to_dict('records'),
        'strength_areas': {category: score for category, score in strength_categories.items() if not pd.isna(score)},
        'top_subjects': df.head(5)['subject'].tolist()
    }

# Function to generate recommendations using LLM
def generate_recommendations(student_data, subject_analysis):
    # Initialize LLM and embeddings
    if not st.session_state.api_key_set:
        return {"error": "API key not set. Please configure your API key first."}
    
    try:
        # Select the appropriate model based on user choice
        if "GPT-3.5" in llm_option:
            llm = ChatOpenAI(temperature=0.7, model="gpt-3.5-turbo")
        elif "GPT-4" in llm_option:
            llm = ChatOpenAI(temperature=0.7, model="gpt-4")
        elif "Claude" in llm_option:
            # Would need to import the appropriate class for Claude
            from langchain.chat_models import ChatAnthropic
            llm = ChatAnthropic(temperature=0.7, model="claude-2")
        else:  # Grok
            # Placeholder for Grok implementation
            # This would need to be replaced with actual Grok implementation
            llm = ChatOpenAI(temperature=0.7, model="gpt-3.5-turbo")
        
        embeddings = OpenAIEmbeddings()
        
        # Create a prompt template
        template = """
        You are an academic and career advisor specializing in recommendations for Pakistani students. 
        You have the following information about a student:
        
        Student Name: {student_name}
        GPA/CGPA: {gpa}
        Top Performing Subjects: {top_subjects}
        Strength Areas: {strength_areas}
        
        Based on this information, please provide:
        
        1. DEGREE RECOMMENDATIONS: Recommend 3-5 specific degree programs well-suited to this student's academic strengths. Focus on programs available in Pakistani universities.
        
        2. SCHOLARSHIP OPPORTUNITIES: List 3-5 relevant scholarship opportunities for Pakistani students in these fields, including both domestic and international options. Include eligibility criteria and application processes.
        
        3. UNIVERSITY RECOMMENDATIONS: Suggest 3-5 specific Pakistani universities that offer excellent programs in the recommended fields. Include admission requirements, fee structures, and application deadlines.
        
        4. CAREER PROSPECTS: Outline potential career paths for each recommended degree, focusing on job opportunities, salary expectations, and growth potential specifically in the Pakistani market.
        
        5. FURTHER DEVELOPMENT: Suggest ways the student could enhance their profile for these programs, including specific extracurricular activities, certifications, or skills development.
        
        Please be specific, realistic, and tailored to the Pakistani educational context and job market. Provide actionable advice that considers the economic and social realities of Pakistan.
        """
        
        prompt = PromptTemplate(
            input_variables=["student_name", "gpa", "top_subjects", "strength_areas"],
            template=template
        )
        
        # Prepare the inputs
        inputs = {
            "student_name": student_data["student_name"],
            "gpa": student_data["cgpa"] if student_data["cgpa"] != "Not found" else student_data["gpa"],
            "top_subjects": ", ".join(subject_analysis["top_subjects"]),
            "strength_areas": str(subject_analysis["strength_areas"])
        }
        
        # Generate the recommendations
        chain = prompt | llm
        recommendations = chain.invoke(inputs)
        
        return {"recommendations": recommendations.content}
    
    except Exception as e:
        return {"error": str(e)}

# Main app layout
st.markdown('<h2 class="sub-header">Upload Student Transcript</h2>', unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="section">', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("Upload a transcript (PDF or image)", type=["pdf", "jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        if uploaded_file.type == "application/pdf":
            extracted_text = extract_text_from_pdf(uploaded_file)
        else:
            st.error("Image-based transcripts require OCR functionality. Please upload a PDF transcript.")
            extracted_text = None
        
        if extracted_text:
            with st.expander("View Extracted Text"):
                st.text(extracted_text)
            
            st.session_state.extracted_data = extract_transcript_data(extracted_text)
            
            # Display extracted information
            col1, col2 = st.columns(2)
            with col1:
                st.subheader("Student Information")
                st.write(f"**Name:** {st.session_state.extracted_data['student_name']}")
                st.write(f"**GPA/CGPA:** {st.session_state.extracted_data['cgpa']}")
            
            with col2:
                st.subheader("Subjects & Grades")
                subject_data = pd.DataFrame({
                    "Subject": st.session_state.extracted_data['subjects'],
                    "Grade": st.session_state.extracted_data['grades']
                })
                st.dataframe(subject_data)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Analysis section
if st.session_state.extracted_data:
    st.markdown('<h2 class="sub-header">Academic Analysis & Recommendations</h2>', unsafe_allow_html=True)
    
    with st.container():
        st.markdown('<div class="section">', unsafe_allow_html=True)
        
        if st.button("Generate Recommendations"):
            with st.spinner("Analyzing transcript and generating personalized recommendations..."):
                # Analyze subject strengths
                subject_analysis = analyze_subject_strengths(st.session_state.extracted_data)
                
                # Generate recommendations
                recommendations = generate_recommendations(st.session_state.extracted_data, subject_analysis)
                
                st.session_state.analysis_results = {
                    "subject_analysis": subject_analysis,
                    "recommendations": recommendations
                }
        
        if st.session_state.analysis_results:
            # Display subject strengths
            st.subheader("Academic Strengths Analysis")
            
            strength_data = st.session_state.analysis_results["subject_analysis"]["strength_areas"]
            strength_df = pd.DataFrame({
                "Category": list(strength_data.keys()),
                "Score": list(strength_data.values())
            })
            
            col1, col2 = st.columns([2, 3])
            
            with col1:
                st.dataframe(strength_df)
            
            with col2:
                # Create a simple bar chart
                st.bar_chart(strength_df.set_index("Category"))
            
            # Display recommendations
            st.subheader("Personalized Recommendations")
            
            if "error" in st.session_state.analysis_results["recommendations"]:
                st.error(st.session_state.analysis_results["recommendations"]["error"])
            else:
                st.markdown(st.session_state.analysis_results["recommendations"]["recommendations"])
        
        st.markdown('</div>', unsafe_allow_html=True)

# Additional resources section
st.markdown('<h2 class="sub-header">Additional Resources</h2>', unsafe_allow_html=True)

with st.container():
    st.markdown('<div class="section">', unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### Pakistani Universities")
        st.markdown("""
        - [Higher Education Commission](https://www.hec.gov.pk/)
        - [National University of Sciences and Technology](https://nust.edu.pk/)
        - [Lahore University of Management Sciences](https://lums.edu.pk/)
        - [University of the Punjab](https://pu.edu.pk/)
        - [Quaid-i-Azam University](https://qau.edu.pk/)
        """)
    
    with col2:
        st.markdown("### Scholarship Resources")
        st.markdown("""
        - [HEC Scholarships](https://www.hec.gov.pk/english/scholarshipsgrants/Pages/default.aspx)
        - [Fulbright Pakistan](https://www.usefpakistan.org/)
        - [British Council Pakistan](https://www.britishcouncil.pk/)
        - [DAAD Pakistan](https://www.daad.de/en/information-services-for-higher-education-institutions/further-information-on-daad-programmes/pakistan/)
        - [Pakistan Education Foundation](https://pef.edu.pk/)
        """)
    
    with col3:
        st.markdown("### Career Resources")
        st.markdown("""
        - [Rozee.pk](https://www.rozee.pk/)
        - [Pakistan Jobs Bank](https://www.pakistanjobsbank.com/)
        - [LinkedIn Pakistan](https://www.linkedin.com/showcase/linkedin-pakistan/)
        - [Pakistan Institute of Management](https://www.pim.com.pk/)
        - [Career Development Centers](https://navttc.gov.pk/)
        """)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Footer
st.markdown("""
<div style="text-align: center; margin-top: 2rem; padding: 1rem; background-color: #F3F4F6; border-radius: 0.5rem;">
    <p>© 2025 Academic & Career Advisor for Pakistani Students</p>
    <p>This application is designed to provide guidance based on academic performance. Always consult with professional academic advisors for final decisions.</p>
</div>
""", unsafe_allow_html=True)
