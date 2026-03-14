#week 19 day 2
#text summerizer - transformers

from transformers import pipeline

summerizer = pipeline("summarization",
                      model="sshleifer/distilbart-cnn-6-6",
                      )

text = """
### Me 🌟📚💼
Rajitha Pathipaka, based in Hyderabad, Telangana, is a dedicated professional with a B.Tech from Jyotishmathi Institute of Technological Sciences and a Certified Data Analyst from Datamites Hyderabad. With experience as a Transaction Associate II at Concentrix from March 2020 to February 2023, I have honed my skills in delivering exceptional customer service, resolving complex issues, and working independently in fast-paced environments.

### Projects 🚀🔍📊
#### Project on Medical Data History 🩺💻
- **Objective:** Develop a user-friendly system using SQL to manage and analyze medical records.
- **Details:** Engineered a secure database for storing patient data, integrated SQL queries for easy access by healthcare professionals, and developed SQL-based tools for data analysis, aiding in informed decision-making for patient care.

#### HR Analytics Initiative 📈👥💡
- **Objective:** Improve workforce management using MS Excel and Tableau.
- **Details:** Conducted data cleansing and modeling in Excel to analyze employee performance, retention, and recruitment trends. Utilized Tableau for visualization, creating interactive dashboards for monitoring key HR metrics. Contributed to informed decision-making by optimizing recruitment strategies and enhancing employee satisfaction through data-driven insights.

### Technical Skills and Others 🛠️💡🎨
- **Data Analysis:** Proficient in SQL, Python, and MS Excel.
- **Data Visualization:** Skilled in Power BI, Tableau, and MS Excel for creating visually appealing dashboards.
- **Problem Solving:** Strong attention to detail and analytical mindset.
- **Customer Service:** Committed to delivering exceptional service and resolving complex issues promptly.

### Achievements & Awards 🏆🥇
- Recognized as "Best Employee" twice.

### Personal Details 📅👤🌍
- **Date of Birth:** July 29, 1997
- **Marital Status:** Single
- **Nationality:** Indian
- **Gender:** Female
- **Languages Known:** English, Hindi, Telugu

"""

summerize = summerizer(
    text,
    max_length=30,
    min_length=10,
    do_sample=False)

print("\nsummerized text:",summerize)