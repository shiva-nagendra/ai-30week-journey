#week 19 day 2
#text summerizer - transformers

from transformers import pipeline

summerizer = pipeline("summarization",
                      model="sshleifer/distilbart-cnn-6-6",
                      )

text = """
 Good knowledge of using find, sed command
Expertise in using Unix Utilities and associated command with Oracle Database
 Scheduling Backup and Backup scripts using CRONTAB Scheduled Jobs.
 Scheduling job by at and crontab feature
 Use a regular expression to search for multiple similar strings
 Hands-on expertise in SQL statements (DDL, DML, DCL, TCL).
 Good Knowledge of all types of constraints.
 Good Knowledge of sequence in SQL
 Working Knowledge of Jira ticketing tool
"""

summerize = summerizer(
    text,
    max_length=60,
    min_length=30,
    do_sample=False)

print("\nsummerized text:",summerize)