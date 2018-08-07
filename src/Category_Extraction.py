# The following code extracts category information from the raw xml documents(golden data) for evalutation purposes
# the code runs as a combination of non-parallel and parallel environments

from xml.dom import minidom
import os
directory = '/home/pasumart/rcv1'
finallist = []

#Required attributes from a individual document is appended to a list

for root,dirs,filenames in os.walk(directory):
        for file in filenames:
            #print(file)
            log = open(os.path.join(root,file),'r')
            doc = minidom.parse(os.path.join(root,file))
            idlist = doc.getElementsByTagName('newsitem')
            for id in idlist:
                value = id.getAttribute("itemid")
            codelist = doc.getElementsByTagName('code')
            words=[]
            for s in codelist:
                words.append(s.attributes['code'].value)
            finallist.append(value)
            finallist.append(words)


			
from pyspark.sql import SparkSession
from pyspark.sql import Row
from pyspark.ml.feature import StopWordsRemover
from pyspark.sql.types import FloatType,IntegerType
import pyspark.sql.functions as psf
from pyspark.sql.functions import *
from pyspark.sql.functions import col,udf
from itertools import islice
from pyspark.sql.types import BooleanType,IntegerType

spark = SparkSession.builder.appName('toExtractTopics.rawxml').getOrCreate()		#Spark process begins from here

df_first = spark.createDataFrame([(x,y) for x,(y) in (zip(
            islice(finallist,0,len(finallist),2),
            islice(finallist,1,len(finallist),2)))],('DocID','Categories'))		#the list obtained is converted to a dataFrame

limit_df = df_first.limit(100000)

#cleaning of the dataFrame to eliminate unwanted information

Filter_regions = StopWordsRemover(inputCol='Categories',outputCol='Topics',stopWords=['AARCT','ABDBI','AFGH','AFRICA','AJMN','ALADI','ALB','ALG','AMSAM',
'ANDEAN','ANDO','ANGOL','ANGUIL','ANTA','ANZUS','APEC','ARABST','ARG','ARMEN','ARMHG','ARUBA','ASEAN','ASIA','AUSNZ','AUST','AUSTR',
'AZERB','BAH','BAHRN','BALTST','BANDH','BARB','BELG','BELZ','BENIN','BENLUX','BERM','BHUTAN','BIOT','BOL','BOTS','BRAZ',
'BRUNEI','BSHZG','BUL','BURMA','BURUN','BVI','BYELRS','CACCM','CAFR','CAM','CAMER','CANA','CANI','CARCOM','CARIB','CASIA',
'CAYI','CEAFR','CHAD','CHIL','CHINA','COL','COMOR','CONGO','COOKIS','COSR','CRTIA','CUBA','CURAC','CVI','CYPR','CZREP','DEN',
'DEVGCO','DIEGO','DOMA','DOMR','DUBAI','EAFR','EASIA','EASTIS','ECOWAS','ECU','EEC','EEUR','EFTA','EGYPT','ELSAL','EQGNA',
'ERTRA','ESTNIA','ETHPA','EUR','EUREA','FAEROE','FALK','FEAST','FESMIC','FGNA','FIJI','FIN','FPDT','FPOLY','FRA','FUJH','GABON',
'GAMB','GCC','GFIVE','GFR','GHANA','GIB','GREECE','GREENL','GREN','GRGIA','GSEVEN','GTEN','GUAD','GUAM','GUAT','GUBI','GULFST',
'GUREP','GUY','HAIT','HKONG','HON','HUNG','ICEL','ICST','INDIA','INDOCH','INDON','INDSUB','IRAN','IRAQ','IRE','ISLAM','ISRAEL',
'ITALY','JAMA','JAP','JORDAN','KAMPA','KAZK','KENYA','KIRB','KIRGH','KUWAIT','LAM','LAOS','LATV','LEBAN','LESOT','LIBER','LIBYA',
'LIECHT','LITH','LUX','MACAO','MAH','MALAG','MALAW','MALAY','MALDR','MALI','MALTA','MARQ','MAURTN','MAURTS','MCDNIA','MEAST','MED',
'MEX','MOLDV','MONAC','MONGLA','MONT','MOROC','MOZAM','MRCSL','NAFR','NAFTA','NAM','NAMIB','NANT','NATO','NAURU','NEPAL','NETH',
'NEWCAL','NICG','NIGEA','NIGER','NIUE','NKOREA','NOMARI','NORFIS','NORW','NZ','OAMS','OAPEC','OAU','OCEANA','OECD','OILEX','OMAN',
'OPEC','PACIS','PACRM','PAKIS','PALAU','PANA','PAPNG','PARA','PERU','PHLNS','PITCIS','POL','PORL','PST','PTAESA','PURI','QATAR',
'RAKH','REUNI','ROM','RUSS','RWANDA','SAARAB','SAARC','SADCC','SAFR','SAM','SASIA','SCAND','SEASIA','SELA','SENEG','SEYCH',
'SHAJH','SILEN','SINGP','SKIT','SKOREA','SLUC','SLVAK','SLVNIA','SMARNO','SOLIL','SOMAL','SOUAFR','SPAIN','SPSAH','SRILAN',
'STHEL','STPM','SUDAN','SURM','SVIN','SWAZD','SWED','SWITZ','SYRIA','TADZK','TAI','TAIWAN','TANZA','TCAI','THAIL','THDWLD',
'TIMOR','TOGO','TOKLAU','TONGA','TRSCUN','TRTO','TUNIS','TURK','TURKM','TVLU','UAE','UAQ','UGANDA','UK','UKRN','UN','UPVOLA',
'URU','USA','USAAK','USAAL','USAAR','USAAZ','USACA','USACO','USACT','USADC','USADE','USAFL','USAGA','USAHI','USAIA','USAID',
'USAIL','USAIN','USAKS','USAKY','USALA','USAMA','USAMD','USAME','USAMI','USAMN','USAMO','USAMS','USAMT','USANC','USAND',
'USANE','USANH','USANJ','USANM','USANV','USANY','USANYC','USAOK','USAOR','USAPA','USARI','USASC','USASD','USATN','USATX',
'USAUT','USAVA','USAVT','USAWA','USAWI','USAWV','USAWY','USSR','UZBK','VANU','VCAN','VEN','VI','VIETN','WAFR','WALLIS','WASIA',
'WEEC','WEIND','WESTW','WEUR','WORLD','WSOMOA','YEMAR','YUG','ZAIRE','ZAMBIA','ZIMBAB'])

Filter = Filter_regions.transform(limit_df)

Result_df = Filter.select('DocID','Topics').repartition(4)

def inter(a,b):
    match = set(a).issubset(set(b))
    return match

inter_udf = udf(inter, BooleanType())

def threshold(value):
    if value == True:
        return 1
    else:
        return 0

threshold_udf = udf(threshold,IntegerType())

df1 = Result_df.join(Result_df.alias("Result_df1").select(col("DocID").alias("DocID2"),col("Topics").alias("Topics2")),col("DocID") < col("DocID2"), 'inner')\
							.withColumn('Intersect_Score',inter_udf(col('Topics'),col('Topics2')))\
							.withColumn('True_match',threshold_udf(col('Intersect_Score')))


Final = df1.select('DocID','DocID2','True_match')

Final.write.parquet('/home/pasumart/goldendata_10p5')

spark.stop()