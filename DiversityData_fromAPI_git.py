
"""
Antti Moelsae 10/2018

"""
import pandas as pd
import math
from census import Census
from us import states
import requests
import numpy as np
import pysal
from io import BytesIO
from zipfile import ZipFile
from urllib.request import urlopen
import shapefile
import io
import re
exportFolder = 'path_to//'

# read census table that has column names
tableShells = pd.read_csv('https://www2.census.gov/programs-surveys/acs/summary_file/2016/documentation/user_tools/ACS_5yr_Seq_Table_Number_Lookup.txt', encoding='ISO-8859-1')
tableShells['Line Number'] = tableShells['Line Number'].astype(str)
tableShells['Line Number'] = tableShells['Line Number'].dropna().apply(lambda x: '{0:0>5}'.format(x)).str.split('.', expand=True)[0]  # df.astype(int) doesn't take NAs
tableShells['Unique ID'] = tableShells['Table ID'] + '_' + tableShells['Line Number']

tableNames = list(set(tableShells['Table ID']))

# get column ID and name
tableDict = {}
allTables = pd.DataFrame(columns=[])
for i in tableNames:
    indTable = tableShells.loc[tableShells['Table ID'] == i]
    indTable = indTable.reset_index()
    indTable.loc[0, 'Unique ID'] = indTable.loc[0, 'Table ID']
    indTable = indTable[['Table Title', 'Unique ID']]
    tableDict.update({indTable.loc[0, 'Unique ID']: indTable.loc[0, 'Table Title']})
    allTables = pd.concat([allTables, indTable], axis=1)

# transpose and drop rows that have no data
allTables.columns = allTables.iloc[0]
allTables.drop(allTables.index[[0, 1]], inplace=True)

comboTables = []
table = 'B03002'  # race

columnCodes = allTables[table].dropna()
nameloc = allTables.columns.get_loc(table) - 1
columnNames = allTables.iloc[:, nameloc].dropna()

dataYears = ['1980', '1990', '2000', '2010', '2011', '2012', '2013', '2014', '2015', '2016']

for dataYear in dataYears:
    if int(dataYear) <= 2000:  # downloaded from https://s4.ad.brown.edu/Projects/Diversity/Researcher/LTBDDload/DataList.aspx
        year = dataYear[2:]
        df = pd.read_csv('path_to//LTDB_Std_{0}_fullcount.csv'.format(dataYear), encoding='latin-1')
        df.rename(columns={'TRTID10': 'FIPS', 'POP' + year: 'Total', 'NHWHT' + year: 'White', 'NHBLK' + year: 'Black', 'ASIAN' + year: 'Asian', 'HISP' + year: 'Hispanic'}, inplace=True)
        df['FIPS'] = df['FIPS'].astype(str)
        df = df.loc[df.iloc[:, 0].str.contains(r'^36')]   # NY tracts
        df.loc[:, 'Other'] = df['Total'] - df['White'] - df['Black'] - df['Asian'] - df['Hispanic']
        df = df[['FIPS', 'Total', 'White', 'Black', 'Asian', 'Hispanic']]

    else:
        c = Census("32dd72aa5e814e89c669a4664fd31dcfc3df333d")
        column = c.acs5.get((table + '_001E'), geo={'for': 'tract:*', 'in': 'state:{} county:*'.format(states.NY.fips)}, year=int(dataYear))
        # get FIPSs for NY
        FIPSlist = []
        for j in column:
            FIPSlist.append(str(j['state'] + j['county'] + j['tract']))

        censusTable = pd.DataFrame(FIPSlist)
        censusTable.columns = ['FIPS']
        # get columns
        for i in columnCodes:
            column = c.acs5.get((i + 'E'), geo={'for': 'tract:*', 'in': 'state:{} county:*'.format(states.NY.fips)}, year=int(dataYear))
            valueList = []
            for j in column:
                valueList.append(int(j[i + 'E']))
            censusTable[i] = pd.DataFrame(valueList)

        columnDict = dict(zip(columnCodes, columnNames))
        for k, v in columnDict.items():
            censusTable.rename(columns={k: v}, inplace=True)

        # EDIT TABLE
        df = censusTable.copy()  # make copy - otherwise gives SettingwithCopyWarning and slows down processing
        sumFrom = 7
        sumTo = 10
        df.loc[:, 'Hispanic'] = df.iloc[:, 1] - df.iloc[:, 2]
        df.loc[:, 'Other'] = df.iloc[:, sumFrom:sumTo].sum(axis=1) + df.iloc[:, 5]
        keepcols = [0, 1, 3, 4, 6, 22, 23]
        df = df.iloc[:, keepcols]

    # abbreviate column names to keyword
    cols = list(df.columns.values)
    conditions = ['Black', 'White', 'Asian', 'Hispanic', 'Other', 'Total']
    for i in conditions:
        for y, item in enumerate(cols):
            if i in item:
                cols[y] = i + dataYear[2:]
    df.columns = cols

    pCols = []
    for i in conditions[:-1]:    # discard Total
        for y in df:
            if i in y:
                pCols.append(i + "P" + dataYear[2:])
                df[i + "P" + dataYear[2:]] = df[i + dataYear[2:]] / df['Total' + dataYear[2:]]

    colName = 'RE' + dataYear[2:]
    df[colName] = 0
    for i in pCols:
        for index, row in df.iterrows():
            n = df.loc[index, i]
            try:
                df.loc[index, colName] = df.loc[index, colName] - n * math.log(n, 5)
            except ValueError:
                df[colName] = df[colName] + 0

    keepcols = [0, 1] + list(range(df.columns.get_loc('BlackP{}'.format(dataYear[2:])), len(df.columns)))  # range needs to extend +1
    df = df.iloc[:, keepcols]
    df['Year'] = dataYear

    vars()['raceTable' + dataYear[2:]] = df
    print("race table" + dataYear)


table = 'B19001'  # income

# get FIPSs for NY

url = 'https://download.bls.gov/pub/time.series/cu/cu.data.1.AllItems'  # inflation multipliers
r = requests.get(url, stream=True)
inflDict = {}
for n, line in enumerate(r.iter_lines(), start=1):
    if line:
        inflDict[str(line).split('\\t')[1]] = str(line).split('\\t')[3]

dataYears = ['2010', '2011', '2012', '2013', '2014', '2015', '2016']
for dataYear in dataYears:
    c = Census("32dd72aa5e814e89c669a4664fd31dcfc3df333d")
    column = c.acs5.get((table + '_001E'), geo={'for': 'tract:*', 'in': 'state:{} county:*'.format(states.NY.fips)}, year=int(dataYear))
    FIPSlist = []
    for j in column:
        FIPSlist.append(str(j['state'] + j['county'] + j['tract']))

    columnCodes = allTables[table].dropna()
    nameloc = allTables.columns.get_loc(table) - 1
    columnNames = allTables.iloc[:, nameloc].dropna()

    column = c.acs5.get((table + '_001E'), geo={'for': 'tract:*', 'in': 'state:{} county:*'.format(states.NY.fips)}, year=int(dataYear))
    censusTable = pd.DataFrame(FIPSlist)
    censusTable.columns = ['FIPS']
    # get columns
    for i in columnCodes:
        column = c.acs5.get((i + 'E'), geo={'for': 'tract:*', 'in': 'state:{} county:*'.format(states.NY.fips)}, year=int(dataYear))
        valueList = []
        for j in column:
            valueList.append(int(j[i + 'E']))
        censusTable[i] = pd.DataFrame(valueList)

    columnDict = dict(zip(columnCodes, columnNames))
    for k, v in columnDict.items():
        censusTable.rename(columns={k: v}, inplace=True)

    # EDIT TABLE
    df = censusTable.copy()

    inflAdj = float(inflDict['2016']) / float(inflDict[dataYear])
    df.loc[:, '0_20k_{}'.format(dataYear[2:])] = df.iloc[:, 2:5].sum(axis=1) * inflAdj
    df.loc[:, '20k_40k_{}'.format(dataYear[2:])] = df.iloc[:, 5:9].sum(axis=1) * inflAdj
    df.loc[:, '40k_75k_{}'.format(dataYear[2:])] = df.iloc[:, 9:13].sum(axis=1) * inflAdj
    df.loc[:, '75k_125k_{}'.format(dataYear[2:])] = df.iloc[:, 13:15].sum(axis=1) * inflAdj
    df.loc[:, '125k_up_{}'.format(dataYear[2:])] = df.iloc[:, 15:18].sum(axis=1) * inflAdj

    df.rename(columns={'Total:': 'TotalHH' + dataYear[2:]}, inplace=True)
    keepcols = [0, 1] + list(range(18, len(df.columns)))  # range needs to extend +1

    pCols = []
    for i in keepcols[2:]:    # discard FIPS + Total
        pcol = df.columns[i][:-2] + "P" + dataYear[2:]
        pCols.append(pcol)
        df[pcol] = df[df.columns[i]] / df['TotalHH' + dataYear[2:]]

    colName = 'IE' + dataYear[2:]
    df[colName] = 0
    for i in pCols:
        for index, row in df.iterrows():
            n = df.loc[index, i]
            try:
                df.loc[index, colName] = df.loc[index, colName] - n * math.log(n, 5)  # substracts on each iteration
            except ValueError:
                df[colName] = df[colName] + 0

    # normalization
    colNameN = 'IEN' + dataYear[2:]
    for index, row in df.iterrows():
        n = df.loc[index, colName]
        try:
            df.loc[index, colNameN] = math.cosh(1.5 * n)**5 / math.cosh(1.5 * df[colName].max())**5
        except ValueError:
            pass

    keepcols = [0, 1] + list(range(df.columns.get_loc('0_20k_P{}'.format(dataYear[2:])), len(df.columns)))  # range needs to extend +1
    df = df.iloc[:, keepcols]

    vars()['incomeTable' + dataYear[2:]] = df
    print("income table" + dataYear)


# combine and calculate composite entropy
comboTables = []
dataYears = ['1980', '1990', '2000', '2010', '2011', '2012', '2013', '2014', '2015', '2016']
for dataYear in dataYears:  # changes to copy are reflected in dataframe
    if int(dataYear) > 2000:
        vars()['comboTable' + dataYear[2:]] = pd.concat([vars()['raceTable' + dataYear[2:]], vars()['incomeTable' + dataYear[2:]]], axis=1)
        vars()['comboTable' + dataYear[2:]].loc[:, 'CE{}'.format(dataYear[2:])] = (vars()['comboTable' + dataYear[2:]].loc[:, 'RE{}'.format(dataYear[2:])] + vars()['comboTable' + dataYear[2:]].loc[:, 'IEN{}'.format(dataYear[2:])]) / 2
        vars()['comboTable' + dataYear[2:]] = vars()['comboTable' + dataYear[2:]].T.drop_duplicates().T

    else:

        vars()['comboTable' + dataYear[2:]] = vars()['raceTable' + dataYear[2:]]
        vars()['comboTable' + dataYear[2:]].loc[:, 'CE{}'.format(dataYear[2:])] = 0
        vars()['comboTable' + dataYear[2:]].loc[:, 'IE{}'.format(dataYear[2:])] = 0

    vars()['comboTable' + dataYear[2:]].to_csv(exportFolder + '/' + 'comboTable' + dataYear[2:] + '.csv')

    comboTables.append('comboTable' + dataYear[2:])


# spatial data https://glenbambrick.com/2015/08/09/prj/
def getWKT_PRJ(epsg_code):
    response = urlopen("http://spatialreference.org/ref/epsg/{0}/prettywkt/".format(epsg_code)).read()     # .read() to get string
    wkt = response.decode('utf-8')
    remove_spaces = wkt.replace(" ", "")
    output = remove_spaces.replace("\n", "")
    return output

# for NYS
# get tracts shapefile for matrix
url = "http://www2.census.gov/geo/tiger/GENZ2010/gz_2010_36_140_00_500k.zip"
resp = urlopen(url)
zf = ZipFile(BytesIO(resp.read()))
zf.namelist()

filenames = [y for y in sorted(zf.namelist()) for ending in ['dbf', 'prj', 'shp', 'shx'] if y.endswith(ending)]  # http://andrewgaidus.com/Reading_Zipped_Shapefiles/
dbf, prj, shp, shx = [io.BytesIO(zf.read(filename)) for filename in filenames]
shape = shapefile.Reader(shp=shp, shx=shx, dbf=dbf)
shape.numRecords
# Create a new shapefile in memory
w = shapefile.Writer(shape.shapeType)

# Copy over the existing fields and geometry
w.fields = shape.fields
w._shapes.extend(shape.shapes())

w.save(exportFolder + "tracts")

for comboTable in comboTables:
    dataYear = eval(comboTable)['Year'].iloc[0]     # .rstrip('0').rstrip('.')    for imported tables
    data_df = eval(comboTable).copy()
#    data_df.drop(data_df.columns[0], axis=1, inplace = True)    # needed for imported tables from csv
    data_df['FIPS'] = '1400000US' + data_df['FIPS'].astype(str)   # this change is done to original table as well if not copy
    shp_df = pd.DataFrame(shape.records())
    shp_df.columns = ['FIPS' if x == 1 else x for x in shp_df.columns]
    df = pd.merge(shp_df, data_df, how='left', on='FIPS')
    # http://pysal.readthedocs.io/en/latest/users/tutorials/autocorrelation.html#local-moran-s-i
    W = pysal.queen_from_shapefile(exportFolder + "tracts" + ".shp")         # is shorter than census tables!!!

    string = ', '.join(str(e) for e in list(df.columns))
    ent_cols = re.findall(r"RE\d\d|CE\d\d|IE\d\d", string)
    for col in ent_cols:
        y = np.array(df[col], dtype=float)
        y = np.nan_to_num(y)
        np.random.seed(12345)
        lm = pysal.Moran_Local(y, W)
        quadrants = []
        i = 0
        for val in lm.p_sim:
            if val < 0.05:
                quadrants.append(lm.q[i])
            else:
                quadrants.append(0)
            i += 1
        df[col + '_Moran'] = lm.Is
        df[col + '_Sig'] = lm.p_sim
        df[col + '_Quad'] = quadrants

    shape = shapefile.Reader(shp=shp, shx=shx, dbf=dbf)     # needs to be reset
    w = shapefile.Writer(shape.shapeType)

    # Copy over the existing fields and geometry
    w.fields = shape.fields
    w._shapes.extend(shape.shapes())

    len_w = len(w.fields)
    for col in df.columns[len_w:]:     # appends column names less the ones already in shapefile
        w.field(col, "F", 10, 5)       # this also adds the fields to any shapefile variable!!!!

    index = 0
    for rec in shape.records():
        recnew = rec + df.iloc[index][len_w:].values.tolist()       # need to either remove first empty column from df or do it like this - the deletion flag field is added to df but not if you append record directly from shape
        w.records.append(recnew)
        index += 1

    exportshp = exportFolder + "Diversity_NYS" + dataYear
    w.save(exportshp)

    prj = open(exportshp + '.prj', "w")
    epsg = getWKT_PRJ("4326")
    prj.write(epsg)
    prj.close()

# for NYC
comboTables = []
dataYears = ['1980', '1990', '2000', '2010', '2011', '2012', '2013', '2014', '2015', '2016']
for dataYear in dataYears:
    if int(dataYear) > 2000:
        vars()['comboTable' + dataYear[2:]] = pd.concat([vars()['raceTable' + dataYear[2:]], vars()['incomeTable' + dataYear[2:]]], axis=1)

        vars()['comboTable' + dataYear[2:]].loc[:, 'CE{}'.format(dataYear[2:])] = (vars()['comboTable' + dataYear[2:]].loc[:, 'RE{}'.format(dataYear[2:])] + vars()['comboTable' + dataYear[2:]].loc[:, 'IEN{}'.format(dataYear[2:])]) / 2

        vars()['comboTable' + dataYear[2:]] = vars()['comboTable' + dataYear[2:]].T.drop_duplicates().T

    else:

        vars()['comboTable' + dataYear[2:]] = vars()['raceTable' + dataYear[2:]]
        vars()['comboTable' + dataYear[2:]].loc[:, 'CE{}'.format(dataYear[2:])] = 0
        vars()['comboTable' + dataYear[2:]].loc[:, 'IE{}'.format(dataYear[2:])] = 0

    vars()['comboTable' + dataYear[2:]].to_csv(exportFolder + '/' + 'comboTable' + dataYear[2:] + '.csv')

    comboTables.append('comboTable' + dataYear[2:])


# get tracts shapefile for matrix
url = "https://www1.nyc.gov/assets/planning/download/zip/data-maps/open-data/nyct2010_18a.zip"
resp = urlopen(url)
zf = ZipFile(BytesIO(resp.read()))
zf.namelist()

filenames = [y for y in sorted(zf.namelist()) for ending in ['dbf', 'prj', 'shp', 'shx'] if y.endswith(ending)]  # http://andrewgaidus.com/Reading_Zipped_Shapefiles/
dbf, prj, shp, shx = [io.BytesIO(zf.read(filename)) for filename in filenames]
shape = shapefile.Reader(shp=shp, shx=shx, dbf=dbf)
shape.numRecords
# Create a new shapefile in memory
w = shapefile.Writer(shape.shapeType)

# Copy over the existing fields and geometry
w.fields = shape.fields
w._shapes.extend(shape.shapes())

w.save(exportFolder + "tracts")

allLISA_df = pd.DataFrame([[]])
allLISA_df_noparks = pd.DataFrame([[]])

for comboTable in comboTables:
    dataYear = eval(comboTable)['Year'].iloc[0]     # if working from csv import .rstrip('0').rstrip('.')
    data_df = eval(comboTable)
    data_df['FIPS'] = data_df['FIPS'].astype(str).replace('^36061', '1', regex=True)
    data_df['FIPS'] = data_df['FIPS'].astype(str).replace('^36005', '2', regex=True)
    data_df['FIPS'] = data_df['FIPS'].astype(str).replace('^36047', '3', regex=True)
    data_df['FIPS'] = data_df['FIPS'].astype(str).replace('^36081', '4', regex=True)
    data_df['FIPS'] = data_df['FIPS'].astype(str).replace('^36085', '5', regex=True)
    shp_df = pd.DataFrame(shape.records())
    shp_df.columns = ['FIPS' if x == 4 else x for x in shp_df.columns]
    shp_df.columns = ['PUMA' if x == 7 else x for x in shp_df.columns]

    df = pd.merge(shp_df, data_df, how='left', on='FIPS')
    df.drop(df.columns[0], axis=1, inplace=True)   # drop empty

    # calculate diversity
    # http://pysal.readthedocs.io/en/latest/users/tutorials/autocorrelation.html#local-moran-s-i
    W = pysal.queen_from_shapefile(exportFolder + "tracts.shp")         # NYC DCP tracts - is shorter than census tables!!!

    string = ', '.join(str(e) for e in list(df.columns))
    ent_cols = re.findall(r"RE\d\d|CE\d\d|IE\d\d", string)
    for col in ent_cols:
        y = np.array(df[col], dtype=float)
        y = np.nan_to_num(y)
        np.random.seed(12345)
        lm = pysal.Moran_Local(y, W)
        quadrants = []
        i = 0
        for val in lm.p_sim:
            if val < 0.05:
                quadrants.append(lm.q[i])
            else:
                quadrants.append(0)
            i += 1
        df[col + '_Moran'] = lm.Is
        df[col + '_Sig'] = lm.p_sim
        df[col + '_Quad'] = quadrants

    PUMA = df['PUMA'].tolist()  # to filter out park tracts
    new_PUMA = []
    for i in PUMA:
        if i.startswith('park'):
            new_PUMA.append('park')
        else:
            new_PUMA.append(i)

    df['PUMA'] = new_PUMA

# shapefiles for with and without parks
    shape = shapefile.Reader(shp=shp, shx=shx, dbf=dbf)
    w = shapefile.Writer(shape.shapeType)

    # Copy over the existing fields and geometry
    w.fields = shape.fields[1:]
    w._shapes.extend(shape.shapes())

    len_w = len(w.fields)
    for col in df.columns[len_w:]:     # appends column names less the ones already in shapefile
        w.field(col, "F", 10, 5)        # this also adds the fields to shp!!! needs to be string, otherwise "cannot convert NaN to integer" error at save
    shape = shapefile.Reader(shp=shp, shx=shx, dbf=dbf)    # this variable needs to be re-created

    for index in range(0, len(shape.shapes())):
        recnew = df.iloc[index].values.tolist()
        w.records.append(recnew)
        index += 1

    exportshp = exportFolder + "Diversity_NYC_" + dataYear
    w.save(exportshp)

    prj = open(exportshp + ".prj", "w")
    epsg = getWKT_PRJ("2263")
    prj.write(epsg)
    prj.close()

    # no parks
    rm_parks_index = df.loc[((df['PUMA'] == 'park') & (df['Total' + dataYear[-2:]] < 100)) | (df['PUMA'] == 'Airport')].index.values.tolist()
    df_noparks = df.drop(df.index[rm_parks_index])

    e = shapefile.Editor(exportFolder + "tracts")       # has only shapes and fields, no records
    n = 0
    for i in rm_parks_index:
        e.delete(i - n)         # for each iteration -1 since not dropping indexes, but nth row
        n += 1

    shape_noparks = shapefile.Reader(e.save())
    e.save(exportFolder + 'tracts_noparks')

    # slight difference in clusters to GeoDa -- centroids are different in GeoDa https://github.com/GeoDaCenter/geoda/issues/1533
    W = pysal.queen_from_shapefile(exportFolder + "tracts_noparks" + ".shp")

    string = ', '.join(str(e) for e in list(df_noparks.columns))
    ent_cols = re.findall(r"RE\d\d|CE\d\d|IE\d\d", string)
    for col in ent_cols:
        y = np.array(df_noparks[col], dtype=float)
        y = np.nan_to_num(y)
        np.random.seed(12345)
        lm = pysal.Moran_Local(y, W)
        quadrants = []
        i = 0
        for val in lm.p_sim:
            if val < 0.05:
                quadrants.append(lm.q[i])
            else:
                quadrants.append(0)
            i += 1
        df_noparks[col + '_Moran'] = lm.Is
        df_noparks[col + '_Sig'] = lm.p_sim
        df_noparks[col + '_Quad'] = quadrants

    year = dataYear[2:]
    df_select = df[['FIPS', 'Year', 'RE' + year + '_Quad', 'IE' + year + '_Quad', 'CE' + year + '_Quad']]
    allLISA_df = pd.concat([allLISA_df, df_select], axis=1)   # one table with all years
    df_select_noparks = df_noparks[['FIPS', 'Year', 'RE' + year + '_Quad', 'IE' + year + '_Quad', 'CE' + year + '_Quad']]
    allLISA_df_noparks = pd.concat([allLISA_df_noparks, df_select_noparks], axis=1)
    print(len(df_select_noparks))

    w = shapefile.Writer(shape_noparks.shapeType)

    # Copy over the existing fields and geometry
    w.fields = shape_noparks.fields
    w._shapes.extend(shape_noparks.shapes())

    len_w = len(w.fields)
    for col in df_noparks.columns[len_w - 1:]:     # appends column names less the ones already in shapefile
        w.field(col, "F", 10, 5)             # this also adds the fields to shp!!!! needs to be string, otherwise "cannot convert NaN to integer" error at save

    shape_noparks = shapefile.Reader(w.save())

    for index in range(0, len(shape_noparks.shapes())):
        recnew = df_noparks.iloc[index].values.tolist()
        w.records.append(recnew)
        index += 1

    exportshp = exportFolder + "Diversity_NYC_noparks_" + dataYear
    w.save(exportshp)

    prj = open(exportshp + ".prj", "w")
    epsg = getWKT_PRJ("2263")
    prj.write(epsg)
    prj.close()

allLISA_df.to_csv(exportFolder + 'allLISA.csv', sep=', ', encoding='utf-8')
allLISA_df_noparks.to_csv(exportFolder + 'allLISA_noparks.csv', sep=', ', encoding='utf-8')
