import numpy as np
import csv
import pandas as pd

def readFile(fileName):
    file = pd.read_csv(fileName)
    return file

def calcPrevOccurences(patientIds, dateOfAtttempt, file, date, columnName):
    prevVisits = []
    for x, y in zip(patientIds, dateOfAtttempt):
        prev = []
        attempt = y.split('-')
        if(len(attempt)==3):
            for w,z in zip(file.get_group(x)[columnName].values,file.get_group(x)[date].values):
                prevDate = z.split('-')
                if(len(prevDate)==3):
                    prev.append([w,z])
        prevVisits.append( prev)
    return prevVisits

def filterBySuicideTendancy(file,columnName, date):
    print(columnName)
    currentFile = file.groupby(['PATIENT'])
    file[columnName] = file[columnName].str.lower()
    file.dropna(subset = [columnName])
    file = file[file[columnName].str.contains('sui', na=False ) ]
    
    patientIds = list(file["PATIENT"].values)
    dateOfAtttempt = list(file[date].values)

    prevVisits = calcPrevOccurences(patientIds, dateOfAtttempt, currentFile, date, columnName)

    return patientIds, dateOfAtttempt, prevVisits

def pickRandomData(file,columnName,date):
    currentFile = file.groupby(['PATIENT'])
    file = file[~file[columnName].str.contains('sui', na=False )]
    patientIds = list(file["PATIENT"].values)
    dateOfVisit = list(file[date].values)
    prevVisits = calcPrevOccurences(patientIds, dateOfVisit, currentFile, date, columnName)
    return patientIds, dateOfVisit, prevVisits



## not needed

## needed

abcd = 1

csvFileName = 'csv'+str(abcd)

allergiesFile = readFile('./CSV/' + csvFileName +'/allergies.csv')
immunizationsFile = readFile('./CSV/' + csvFileName +'/immunizations.csv')
medicationsFile = readFile('./CSV/' + csvFileName +'/medications.csv')
observationsFile = readFile('./CSV/' + csvFileName +'/observations.csv')

carePlansFile= readFile('./CSV/' + csvFileName +'/careplans.csv')
conditionsFile = readFile('./CSV/' + csvFileName +'/conditions.csv')
encountersFile = readFile('./CSV/' + csvFileName +'/encounters.csv')
proceduresFile = readFile('./CSV/' + csvFileName +'/procedures.csv')



patientsColumnNames = ["ID","BIRTHDATE","DEATHDATE","SSN","DRIVERS","PASSPORT","PREFIX","FIRST","LAST","SUFFIX","MAIDEN","MARITAL","RACE","ETHNICITY","GENDER","BIRTHPLACE","ADDRESS",'SUICIDE_AGE','SUICIDE_YEAR','PREV_ENCOUNTERS']

patientIdListForSuicide = []
datesList = []
prevOccurences = []

def appendToExistingData(file,columnName,date):
    global patientIdListForSuicide
    global datesList
    global prevOccurences

    ids, dates, prevVisits = filterBySuicideTendancy(file, columnName,date)
    patientIdListForSuicide+= ids
    datesList += dates
    prevOccurences+=prevVisits

def generate_positive_samples():
    global prevOccurences

    appendToExistingData(carePlansFile, "DESCRIPTION",'START')
    appendToExistingData(carePlansFile, "REASONDESCRIPTION", 'START')
    print("Done with 1")

    appendToExistingData(conditionsFile, "DESCRIPTION", 'START')
    print("Done with 2")

    appendToExistingData(encountersFile, "DESCRIPTION", 'DATE')
    appendToExistingData(encountersFile, "REASONDESCRIPTION", 'DATE')
    print("Done with 3")

    appendToExistingData(proceduresFile, "DESCRIPTION",'DATE')
    appendToExistingData(proceduresFile, "REASONDESCRIPTION",'DATE')
    print("Done with 4")

    appendToExistingData(immunizationsFile, "DESCRIPTION",'DATE')
    print("Done with 5")

    appendToExistingData(observationsFile, "DESCRIPTION",'DATE')
    print("Done with 6")

    appendToExistingData(medicationsFile, "DESCRIPTION",'START')
    appendToExistingData(medicationsFile, "REASONDESCRIPTION",'START')
    print("Done with 7")

    appendToExistingData(allergiesFile, "DESCRIPTION",'START')
    print("Done with 8")

    a = np.unique(patientIdListForSuicide, return_index=True)

    finalPatientIds = np.take(patientIdListForSuicide, a[1])
    finalDateList = np.take(datesList, a[1])
    prevOccurences = np.take(prevOccurences,a[1])

    dictionary = {}
    prevOccurencesDictionary = {}
    for x,y,z in zip(finalDateList, finalPatientIds, prevOccurences):
        dictionary[y] = x
        prevOccurencesDictionary[y] = z

    patientRecords = []
    with open('./CSV/' + csvFileName +'/patients.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        count = 0
        for row in readCSV:
            if(row and len(row)==17 and row[0]!="" and row[0] in finalPatientIds):
                birthYear = int(row[1].split('-')[0])##,int(row[1].split('-')[1]),int(row[1].split('-')[2])
                suicideYear = int(dictionary[row[0]].split('-')[0])
                row.append(str(suicideYear-birthYear))
                row.append(dictionary[row[0]].split('-'))
                
                prevOccurences = prevOccurencesDictionary[row[0]]
                newListOfOccurences = []
                for x in prevOccurences:
                    reason, time = x
                    # print(reason,time)
                    # input()
                    visitYear = int(time.split('-')[0])
                    # visitYear, visitMonth, visitDay = int(time.split('-')[0]),int(time.split('-')[1]),int(time.split('-')[2])
                    # if(visitYear<birthYear and visitMonth < birthMonth and visitDay < birthDate):
                    #     continue
                    newListOfOccurences.append([reason, str(visitYear-birthYear)])
                row.append(newListOfOccurences)
                patientRecords.append(row)

    print(len(patientRecords))

    patientRecordsDf = pd.DataFrame(data=patientRecords, columns=patientsColumnNames)
    print(patientRecordsDf.head(100))

    patientRecordsDf.to_csv("./Modified Outputs/Modified Output "+str(abcd))

patientIdListForOthers = []
datesListForOthers = []
prevOccurences = []

# generate_positive_samples()

def appendNegativeSamples(file, columnName, date):
    global patientIdListForOthers
    global datesListForOthers
    global prevOccurences

    ids, dates, prevVisits = pickRandomData(file, columnName,date)
    patientIdListForOthers+= ids
    datesListForOthers += dates
    prevOccurences+=prevVisits


def generate_negative_samples():
    global prevOccurences

    appendNegativeSamples(carePlansFile,  "DESCRIPTION",'START')
    appendNegativeSamples(carePlansFile, "REASONDESCRIPTION", 'START')
    print("Done with 1")
    
    appendNegativeSamples(conditionsFile, "DESCRIPTION", 'START')
    print("Done with 2")

    appendNegativeSamples(encountersFile, "DESCRIPTION", 'DATE')
    appendNegativeSamples(encountersFile, "REASONDESCRIPTION", 'DATE')
    print("Done with 3")


    appendNegativeSamples(proceduresFile, "DESCRIPTION",'DATE')
    appendNegativeSamples(proceduresFile, "REASONDESCRIPTION",'DATE')
    print("Done with 4")

    appendNegativeSamples(immunizationsFile, "DESCRIPTION",'DATE')
    print("Done with 5")

    appendNegativeSamples(observationsFile, "DESCRIPTION",'DATE')
    print("Done with 6")

    appendNegativeSamples(medicationsFile, "DESCRIPTION",'START')
    appendNegativeSamples(medicationsFile, "REASONDESCRIPTION",'START')
    print("Done with 7")

    appendToExistingData(allergiesFile, "DESCRIPTION",'START')
    print("Done with 8")

    a = np.unique(patientIdListForOthers, return_index=True)

    finalPatientIds = np.take(patientIdListForOthers, a[1])
    finalDateList = np.take(datesListForOthers, a[1])
    prevOccurences = np.take(prevOccurences,a[1])

    dictionary = {}

    prevOccurencesDictionary = {}
    for x,y,z in zip(finalDateList, finalPatientIds, prevOccurences):
        dictionary[y] = x
        prevOccurencesDictionary[y] = z

    patientRecords = []
    with open('./CSV/' + csvFileName +'/patients.csv') as csvfile:
        readCSV = csv.reader(csvfile, delimiter=',')
        count = 0
        for row in readCSV:
            if(row and len(row)==17 and row[0]!="" and row[0] in finalPatientIds):
                birthYear = int(row[1].split('-')[0])
                suicideYear = int(dictionary[row[0]].split('-')[0])
                row.append(str(suicideYear-birthYear))
                row.append(dictionary[row[0]].split('-'))
                
                prevOccurences = prevOccurencesDictionary[row[0]]
                newListOfOccurences = []
                for x in prevOccurences:
                    reason, time = x
                    visitYear = int(time.split('-')[0])
                    
                    newListOfOccurences.append([reason, str(visitYear-birthYear)])
                row.append(newListOfOccurences)
                patientRecords.append(row)
                
    print(len(patientRecords))

    patientRecordsDf = pd.DataFrame(data=patientRecords, columns=patientsColumnNames)
    print(patientRecordsDf.head(100))

    patientRecordsDf.to_csv("./Negative Samples/neg "+str(abcd))

generate_negative_samples()   