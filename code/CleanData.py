import pandas as pd
import re

# Given the original Relativity data:
#  1, select following columns: Artifact ID','From', 'To', 'Subject', 'First Line Review Decision', 'First Line Review Reason','Extracted Text'
#  2, select rows where "First Line Review Decision" is not null. (extract labelled data)
#  3, select emails whose reason for escalation is not "non-english".
def processRawData(df):

    df = df[['Artifact ID','From', 'To', 'Subject', 'First Line Review Decision', 'First Line Review Reason','Extracted Text']]
    df = df[df['First Line Review Decision'].notnull()]
    df = df[df['First Line Review Reason'] != 'NE (Non-English language communication)']

    print(df.columns)
    print(len(df.index))

    return df


def addFeatures(df):

    df = numberReceiver(df)
    df = personalEmails(df)
    df = posInChain(df)
    df = length(df)                                               # do not change the order of length() and cleanText()
    df = cleanText(df)
    df = concatText(df)

    df = df.drop(['Subject', 'Unnamed: 0', 'From', 'To'], axis=1)
    df = df[df["length"] != 0]
    df = df.rename(columns={"Extracted Text": "text", "First Line Review Decision": "labels"})
    df["labels"].replace({"False Positive": 0, "Escalate to Second line": 1}, inplace=True)

    return df

def numberReceiver(df):
    df["#Receiver"] = df.apply(lambda x: countReceiver(x), axis=1)
    return df

def personalEmails(df):
    df['receiverPers'] = df.apply(lambda x: containPersonal(x, 'To'), axis=1)
    df['senderPers'] = df.apply(lambda x: containPersonal(x, 'From'), axis=1)
    return df

def posInChain(df):
    df['posInChain'] = df.apply(lambda x: countPosInChain(x), axis=1)
    return df

def length(df):
    df['length'] = df.apply(lambda x: countWord(x), axis=1)
    return df

def cleanText(df):
    df['Extracted Text'] = df.apply(lambda x: cleanTextInstance(x), axis=1)
    return df

def concatText(df):
    df['concatText'] = df.apply(lambda x: concatTextInstance(x), axis=1)
    return df
#######################################################################################################################

# Remove the header of the email body
# Remove the forwarding/replying emails before
# Add email subject to text
def cleanTextInstance(row):

    text = str(row["Extracted Text"])
    subject = str(row["Subject"])
    text = text.split("\n", 6)[6]
    text = re.sub(r'From:.*\nSent:[\s\S]*', "", text)
    text = subject + "[SEP]" + text

    return text

# Count words in email body, Subject not included.
# This function should be called before cleanText() --> see function addFeatures()
def countWord(row):
    text = str(row["Extracted Text"])
    text = text.split("\n", 6)[6]
    text = re.sub(r'From:.*\nSent:[\s\S]*', "", text)

    return len(text.split())

def countReceiver(row):
    return str(row['To']).count("@")

# Check whether there is a personal email in sender/receiver list.
def containPersonal(row, col):
    emails = str(row[col])
    personalEmails = ["gmail", "hotmail", "yahoo"]
    for item in personalEmails:
        if item in emails:
            return 1
        else:
            continue
    return 0

# Count how often the email has been forwarded/replied.
def countPosInChain(row):
    text = row['Extracted Text']
    prefixes = re.findall(r'From:.*\nSent:', text)
    return len(prefixes)+1

# Encode the meta data (numercial/categorical features) into text.
# Input data for transformer model (not multimodal transformer).

def concatTextInstance(row):
    emailBody = str(row['Extracted Text'])
    text = "sent"
    num = row['senderPers']

    if (row['senderPers'] == 1):
        text = text + " from a personal email"
    text = text + " to " + str(row['#Receiver']) + " receiver"

    if (row['#Receiver'] > 1):
        text = text + "s"
    if (row['receiverPers'] == 1):
        text = text + ", including personal email addresses"

    text = text + "[SEP]contains " + str(row["length"]) + " words"
    if (row['posInChain'] > 1):
        text = text + "[SEP]forwarded " + str(row['posInChain']) + " times"

    text = text + "[SEP]" + emailBody

    return text

def main():
    data = pd.read_csv("C:\\Users\\YG56QI\\email_monitoring\\data\\data.csv")
    data = addFeatures(data)
    print(data.columns)
    #print(data.dtypes)


    data = data[data["senderPers"] == 1]
    print(len(data.index))

    index = 11
    print(data.iloc[index]["concatText"])
    #print(data.iloc[index]["Extracted Text"])
    #print(data.iloc[index]["First Line Review Reason"])
    #print(data.iloc[index]["senderPers"])
    #print(data.iloc[index]["receiverPers"])
    #print(data.iloc[index]["posInChain"])
    #print(data.iloc[index]["Artifact ID"])
    #print(data.describe())


main()