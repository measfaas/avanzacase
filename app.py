from dash import Dash, html, dcc, callback, Input, Output, State
import plotly.graph_objects as go
import pandas as pd
import dash
import plotly.figure_factory as ff
import dash_bootstrap_components as dbc
import plotly.express as px
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


# Construct the full path to the dataset
filsokvag = "Dataset till case - (2024).csv"

# Read the dataset
data = pd.read_csv(filsokvag, delimiter=';')


# Prepare data
capital_columns = [
    "Totalt kapital på Avanza", "Totalt kapital i Auto", 
    "Kapital i aktier", "Kapital i fonder (inklusive Auto)"
]
auto_columns = [
    "Kapital i Avanza Auto 1", "Kapital i Avanza Auto 2", "Kapital i Avanza Auto 3",
    "Kapital i Avanza Auto 4", "Kapital i Avanza Auto 5", "Kapital i Avanza Auto 6"
]
auto_capital = data[auto_columns].sum().reset_index()
auto_capital.columns = ["Auto-produkter", "Totalt kapital"]
gender_distribution = data['Kön'].value_counts()
age_distribution = data['Åldersintervall'].value_counts()
login_distribution = data["Inloggade dagar senaste månaden"].value_counts().reset_index()
login_distribution.columns = ["Inloggade dagar", "Antal kunder"]
capital_by_gender = data.groupby("Kön")["Totalt kapital på Avanza"].mean().reset_index()
capital_by_age = data.groupby("Åldersintervall")["Totalt kapital på Avanza"].mean().reset_index()




login_activity = data.groupby(['Kön', 'Åldersintervall'])['Inloggade dagar senaste månaden'].mean().unstack()
activity_vs_capital = data.groupby('Kön')[['Totalt kapital på Avanza', 'Inloggade dagar senaste månaden']].mean()
auto_investments = data.groupby('Åldersintervall')[
    ['Kapital i Avanza Auto 1', 'Kapital i Avanza Auto 2', 'Kapital i Avanza Auto 3',
     'Kapital i Avanza Auto 4', 'Kapital i Avanza Auto 5', 'Kapital i Avanza Auto 6']].sum()
data['Auto vs Non-Auto'] = data['Totalt kapital i Auto'] / data['Totalt kapital på Avanza']
auto_preferences = data.groupby(['Kön', 'Åldersintervall'])['Auto vs Non-Auto'].mean().unstack()




# Correlation Matrix
numerical_columns = data.select_dtypes(include='number')
categorical_columns = ["Kön", "Åldersintervall"]
correlation_matrix = numerical_columns.corr()

def remove_outliers(df, columns):
    df_filtered = df.copy()
    for col in columns:
        threshold = df[col].quantile(0.95)
        if threshold > 0:  # Endast ta bort om percentilen är större än 0
            df_filtered = df_filtered[df_filtered[col] <= threshold]
    return df_filtered

data_no_outliers = remove_outliers(data.copy(), numerical_columns)


numerical_columns_no_outliers = data_no_outliers.select_dtypes(include='number')
correlation_matrix_no_outliers = numerical_columns_no_outliers.corr()




app = Dash(__name__)
server = app.server

# Theme and data setup
theme = {
    "background_page": "#f4f4f9",
    "background_content": "#ffffff",
    "primary_color": "#5e60ce",
    "secondary_color": "#6930c3",
    "text_color": "#333333",
    "border_color": "#dddddd",
    "font_family": "Roboto, sans-serif",
    "font_size": "16px",
    "heading_color": "#4a4e69",
}


def create_slide(content, slide_title):
    """Reusable slide layout."""
    return html.Div(
        style={
            "backgroundColor": theme["background_content"],
            "padding": "30px",
            "borderRadius": "10px",
            "boxShadow": "0 4px 6px rgba(0, 0, 0, 0.1)",
            "textAlign": "center",
        },
        children=[
            html.H2(
                slide_title,
                style={
                    "color": theme["heading_color"],
                    "marginBottom": "20px",
                    "fontSize": "24px",
                },
            ),
            content,
        ],
    )

def slide_1():
    return create_slide(html.P("Din uppgift är att beskriva vilken analys du skulle göra för att utvärdera hur det går för Avanza Auto sedan lanseringen. Förklara vad du skulle fokusera på och vilka frågor som du ser som centrala. Det är okej att hitta på siffror för att visa konkreta exempel på hur analysen skulle se ut."), "Utvärdering Avanza Auto")

def slide_2():
    content = html.Div(
        children=[
            html.P(
                "Avanza Auto är en tjänst som erbjuder automatiserad förvaltning av dina investeringar. Genom att använda modern portföljteori och vetenskapliga modeller strävar Avanza Auto efter att optimera dina investeringar baserat på din valda risknivå och sparhorisont. Tjänsten består av sex olika specialfonder, där varje fond representerar en specifik risknivå och förväntad avkastning. Detta gör det enkelt för dig att välja en fond som passar dina individuella sparmål och riskpreferenser."
            ),
            html.P(
                "Avanza Auto lanserades för att möta behovet hos sparare som saknar tid, intresse eller kunskap att själva hantera sina investeringar på bästa sätt. Många som söker hjälp med sitt sparande möts ofta av höga avgifter, vilket Avanza anser är fel. Genom Avanza Auto erbjuder de en automatiserad och kostnadseffektiv förvaltning baserad på modern portföljteori, vilket ger kunderna tillgång till en smart och billig förvaltning av sina pengar"
            ),
        ]
    )
    return create_slide(content, "Avanza Auto? Syfte")

def slide_kpi():
    content = html.Div(
        children=[
            html.H3("Kundrelaterade KPI", style={"marginBottom": "10px", "color": theme["heading_color"]}),
            html.Ul(
                [
                    html.Li("Antal aktiva kunder"),
                    html.Ul([
                        html.Li("Totalt antal kunder som har investerat i Avanza Auto."),
                        html.Li("Tillväxt i antal kunder månad för månad eller år över år."),
                        html.Li("Andel nya kunder som väljer Avanza Auto som första produkt.")
                    ]),
                    html.Li("Genomsnittligt kapital per kund (AUM per kund)"),
                    html.Ul([
                        html.Li("Hur mycket kapital investerar varje kund i genomsnitt?"),
                        html.Li("Trender i insättningar och uttag.")
                    ]),
                    html.Li("Kundsegmentering"),
                    html.Ul([
                        html.Li("Fördelning av kunder baserat på ålder, inkomst, riskprofil och investeringshorisont.")
                    ]),
                    html.Li("Conversion rate"),
                    html.Ul([
                        html.Li("Andelen potentiella kunder (t.ex. de som besöker Avanza Auto-sidan) som faktiskt investerar i produkten.")
                    ]),
                    html.Li("Retention rate"),
                    html.Ul([
                        html.Li("Andelen kunder som fortsätter använda Avanza Auto efter en viss tid (t.ex. 6 eller 12 månader).")
                    ]),
                    html.Li("Supportärenden per kund"),
                    html.Ul([
                        html.Li("Antal supportärenden som relaterar till Avanza Auto per kund.")
                    ]),
                    html.Li("Churn rate"),
                    html.Ul([
                        html.Li("Andelen kunder som lämnar eller avslutar sitt sparande i Avanza Auto.")
                    ])
                ],
                style={"marginBottom": "20px"}
            ),
            html.H3("Finansiella KPI", style={"marginBottom": "10px", "color": theme["heading_color"]}),
            html.Ul(
                [
                    html.Li("Assets Under Management (AUM)"),
                    html.Ul([
                        html.Li("Totalt kapital som förvaltas inom Avanza Auto."),
                        html.Li("Tillväxt i AUM över tid.")
                    ]),
                    html.Li("Avkastning per risknivå"),
                    html.Ul([
                        html.Li("Genomsnittlig årlig avkastning per fond jämfört med respektive jämförelseindex.")
                    ]),
                    html.Li("Kundens kostnad per insatt kapital"),
                    html.Ul([
                        html.Li("Procentuell total avgift för kunden (förvaltningsavgift + underliggande avgifter).")
                    ])
                ],
                style={"marginBottom": "20px"}
            ),
            html.H3("Produktprestanda KPI", style={"marginBottom": "10px", "color": theme["heading_color"]}),
            html.Ul(
                [
                    html.Li("Sharpe-kvot, value at risk (VaR) och andra riskmått"),
                    html.Ul([
                        html.Li("Riskjusterad avkastning för respektive fond.")
                    ]),
                    html.Li("Tracking error"),
                    html.Ul([
                        html.Li("Skillnaden mellan fondens avkastning och dess jämförelseindex.")
                    ]),
                    html.Li("Frekvens av insättningar och uttag"),
                    html.Ul([
                        html.Li("Hur ofta gör kunder insättningar eller uttag?")
                    ])
                ],
                style={"marginBottom": "20px"}
            ),
            html.H3("Marknadsrelaterade KPI", style={"marginBottom": "10px", "color": theme["heading_color"]}),
            html.Ul(
                [
                    html.Li("Marknadsandel inom automatiserad förvaltning"),
                    html.Ul([
                        html.Li("Andelen av marknaden för robotrådgivning eller automatiserade fonder som Avanza Auto innehar.")
                    ])
                ]
            )
        ],
        style={"textAlign": "left", "lineHeight": "1.6"}
    )
    return create_slide(content, "KPI för Avanza Auto")



def slide_recommendation():
    content = html.Div(
        children=[
            html.H3("Rekommendation för analysen", style={"marginBottom": "10px", "color": theme["heading_color"]}),
            html.P(
                "Fokusera först på de KPI som tydligt visar tillväxt, kvalitet i förvaltning och kundnöjdhet:",
                style={"marginBottom": "10px"}
            ),
            html.Ul(
                [
                    html.Li("AUM: För att förstå produktens storlek och tillväxt."),
                    html.Li("Avkastning och riskjusterad avkastning: För att visa prestanda."),
                    html.Li("Retention rate och churn rate: För att mäta kundens långsiktiga engagemang.")
                ],
                style={"marginBottom": "20px"}
            ),
            html.P(
                "Därefter kan du lägga till mer detaljerade KPI (t.ex. kundsegmentering eller hållbarhetsaspekter) för att få en djupare förståelse och identifiera förbättringsområden.",
                style={"marginBottom": "10px"}
            )
        ],
        style={"textAlign": "left", "lineHeight": "1.6"}
    )
    return create_slide(content, "Rekommendation för analysen")

def slide_challenges():
    content = html.Div(
        children=[
            html.H3("Svårigheter och utmaningar", style={"marginBottom": "10px", "color": theme["heading_color"]}),
            html.H4("1. Tidsramar för utvärdering", style={"marginBottom": "5px"}),
            html.Ul(
                [
                    html.Li(
                        "Långsiktighet i sparande: Eftersom Avanza Auto är en produkt som riktar sig till långsiktigt sparande, kan det ta flera år innan den fulla potentialen blir synlig."
                    )
                ],
                style={"marginBottom": "15px"}
            ),
            html.H4("2. Konkurrens och marknadsdynamik", style={"marginBottom": "5px"}),
            html.Ul(
                [
                    html.Li(
                        "Liknande produkter: Konkurrenter som Lysa och Opti erbjuder liknande automatiserade lösningar. Att avgöra om eventuella framgångar beror på Avanza Autos unika egenskaper eller marknadens generella tillväxt kan vara svårt."
                    ),
                    html.Li(
                        "Marknadsposition: Även om Avanza Auto växer kan det vara svårt att avgöra om tillväxten är tillräcklig jämfört med marknadens utveckling."
                    )
                ],
                style={"marginBottom": "15px"}
            ),
            html.H4("3. Kundbeteende och segmentering", style={"marginBottom": "5px"}),
            html.Ul(
                [
                    html.Li(
                        "Bred målgrupp: Avanza Auto riktar sig till både nybörjare och erfarna sparare. Att avgöra hur väl produkten presterar för olika kundsegment kräver djup analys."
                    ),
                    html.Li(
                        "Oberoende faktorer: Kundernas beslut att behålla eller avsluta sitt sparande kan påverkas av externa faktorer, som marknadens utveckling, snarare än produktens kvalitet."
                    )
                ],
                style={"marginBottom": "15px"}
            ),
            html.H4("4. Mångfacetterade prestationsmått", style={"marginBottom": "5px"}),
            html.Ul(
                [
                    html.Li(
                        "Avkastning vs. risk: Kunder kan uppfatta en 'dålig' avkastning som produktens fel, även om den presterar väl i förhållande till marknadens utveckling och risknivå."
                    ),
                    html.Li(
                        "Kundnöjdhet: Kundernas uppfattning om produkten kan skilja sig från dess faktiska finansiella prestanda."
                    )
                ],
                style={"marginBottom": "15px"}
            )
        ],
        style={"textAlign": "left", "lineHeight": "1.6"}
    )
    return create_slide(content, "Svårigheter och utmaningar")

def slide_ml_models():
    content = html.Div(
        children=[
            html.H3("Relevanta maskininlärningsmodeller och tillämpningar", style={"marginBottom": "10px", "color": theme["heading_color"]}),
            html.H4("1. Kundsegmentering och klustring", style={"marginBottom": "5px"}),
            html.Ul(
                [
                    html.Li("Modell: K-means, DBSCAN, eller hierarkisk klustring."),
                    html.Li("Syfte: Identifiera olika kundsegment baserat på demografisk data, insättningsmönster, och riskpreferenser. Detta kan användas för att anpassa marknadsföring och utveckling av produktens funktioner.")
                ],
                style={"marginBottom": "15px"}
            ),
            html.H4("2. Churn-prediktion", style={"marginBottom": "5px"}),
            html.Ul(
                [
                    html.Li("Modell: Random Forest, Gradient Boosting (t.ex. XGBoost), eller Logistic Regression."),
                    html.Li("Syfte: Förutsäga vilka kunder som riskerar att lämna produkten baserat på historisk data om användarbeteende, insättningar/uttag, och marknadsförhållanden.")
                ],
                style={"marginBottom": "15px"}
            )
        ],
        style={"textAlign": "left", "lineHeight": "1.6"}
    )
    return create_slide(content, "Maskininlärningsmodeller och tillämpningar")


def slide_churn_example():
    content = html.Div(
        children=[
            html.H3("Exempel: Enkel implementation av churn-prediktion", style={"marginBottom": "10px", "color": theme["heading_color"]}),
            html.H4("Steg 1: Förbereda data", style={"marginBottom": "5px"}),
            html.Ul(
                [
                    html.Li("Kund-ID"),
                    html.Li("Insättningsmönster (frekvens och volym)"),
                    html.Li("Risknivå (val av fond)"),
                    html.Li("Historik av avkastning"),
                    html.Li("Supportärenden"),
                    html.Li("Aktivitet (inloggningar, uttag)"),
                    html.Li("Churn (0 = stannar, 1 = lämnar)")
                ],
                style={"marginBottom": "15px"}
            ),
            html.H4("Steg 2: Kodexempel i Python", style={"marginBottom": "5px"}),
            html.Pre(
                """
# -*- coding: utf-8 -*-
"
Created on Sun Nov 17 11:10:13 2024

@author: faas
"
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# Ladda och förbered data
import pandas as pd
data = pd.read_csv("customer_data.csv")

# Separera features och target
X = data.drop(columns=["Churn", "CustomerID"])  # Features
y = data["Churn"]  # Target

# Dela upp i träning och testning
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Bygg modellen
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Prediktion
y_pred = rf_model.predict(X_test)

# Utvärdering
print("Classification Report:")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))
                """,
                style={
                    "backgroundColor": "#f4f4f9",
                    "padding": "10px",
                    "borderRadius": "5px",
                    "fontSize": "14px",
                    "overflowX": "scroll",
                    "whiteSpace": "pre-wrap",
                },
            ),
            html.H4("Steg 3: Insikter", style={"marginBottom": "5px"}),
            html.Ul(
                [
                    html.Li("Feature Importance: Modellen kan avslöja vilka faktorer som är viktigast för att förutsäga churn, t.ex. insättningsfrekvens eller supportärenden."),
                    html.Li("Precision och Recall: Utvärdera modellens förmåga att korrekt identifiera kunder som riskerar att lämna.")
                ],
                style={"marginBottom": "15px"}
            ),
            html.H4("Fördelar med avancerad modellering", style={"marginBottom": "5px"}),
            html.Ul(
                [
                    html.Li("Automatisering: Sparar tid genom att analysera stora datamängder effektivt."),
                    html.Li("Prediktion: Möjliggör proaktiva åtgärder, t.ex. riktade erbjudanden till kunder med hög churn-risk."),
                    html.Li("Optimering: Identifierar vilka förändringar som skulle förbättra produkten eller minska churn.")
                ],
                style={"marginBottom": "15px"}
            )
        ],
        style={"textAlign": "left", "lineHeight": "1.6"}
    )
    return create_slide(content, "Churn-prediktion: Exempel")



def slide_customer_analysis():
    content = html.Div(
        children=[
            html.H4("Syfte", style={"marginBottom": "5px"}),
            html.P(
                "Här vill vi i större utsträckning undersöka din förmåga att hantera och analysera data.",
                style={"marginBottom": "15px"}
            ),
            html.H4("Bakgrund", style={"marginBottom": "5px"}),
            html.P(
                "Teamet som ansvarar för Avanza Auto har fått en ny produktägare som vill få en förståelse för kunder som äger Avanza Auto. Din uppgift är att med hjälp av dataunderlaget ge en övergripande bild över Autokunderna.",
                style={"marginBottom": "15px"}
            )
        ],
        style={"textAlign": "left", "lineHeight": "1.6"}
    )
    return create_slide(content, "Analys av Autokunder")

def slide_dataset_analysis():
    content = html.Div(
        children=[
            html.H3("Kod och Dataset: Struktur och Analys", style={"marginBottom": "10px", "color": theme["heading_color"]}),
            html.H4("Kodexempel", style={"marginBottom": "5px"}),
            html.Pre(
                """
import pandas as pd
filsökväg= "C:/temp2/user-faas/avanza/Dataset till case - (2024).csv"

data = pd.read_csv(filsökväg, delimiter=';')

# Visa datasetets struktur
print(data.info())
                """,
                style={
                    "backgroundColor": "#f4f4f9",
                    "padding": "10px",
                    "borderRadius": "5px",
                    "fontSize": "14px",
                    "overflowX": "scroll",
                    "whiteSpace": "pre-wrap",
                },
            ),
            html.H4("Output", style={"marginBottom": "5px"}),
            html.Pre(
                """
<class 'pandas.core.frame.DataFrame'>
RangeIndex: 20000 entries, 0 to 19999
Data columns (total 14 columns):
 #   Column                             Non-Null Count  Dtype 
---  ------                             --------------  ----- 
 0   Totalt kapital på Avanza           20000 non-null  int64 
 1   Totalt kapital i Auto              20000 non-null  int64 
 2   Kapital i Avanza Auto 1            20000 non-null  int64 
 3   Kapital i Avanza Auto 2            20000 non-null  int64 
 4   Kapital i Avanza Auto 3            20000 non-null  int64 
 5   Kapital i Avanza Auto 4            20000 non-null  int64 
 6   Kapital i Avanza Auto 5            20000 non-null  int64 
 7   Kapital i Avanza Auto 6            20000 non-null  int64 
 8   Kapital i aktier                   20000 non-null  int64 
 9   Kapital i fonder (inklusive Auto)  20000 non-null  int64 
 10  Kund sedan år                      20000 non-null  int64 
 11  Kön                                20000 non-null  object
 12  Åldersintervall                    20000 non-null  object
 13  Inloggade dagar senaste månaden    20000 non-null  int64 
dtypes: int64(12), object(2)
memory usage: 2.1+ MB
                """,
                style={
                    "backgroundColor": "#f4f4f9",
                    "padding": "10px",
                    "borderRadius": "5px",
                    "fontSize": "14px",
                    "overflowX": "scroll",
                    "whiteSpace": "pre-wrap",
                },
            )
        ],
        style={"textAlign": "left", "lineHeight": "1.6"}
    )
    return create_slide(content, "1. Kod och Dataset: Struktur och Analys")


def slide_unique_values():
    content = html.Div(
        children=[
            html.H3("Kod och Dataset: Unika värden per kolumn", style={"marginBottom": "10px", "color": theme["heading_color"]}),
            html.H4("Kodexempel", style={"marginBottom": "5px"}),
            html.Pre(
                """
for col in data.columns:
    print(f"{col}: {data[col].unique()[:15]}")  # Visa de första 15 unika värdena
                """,
                style={
                    "backgroundColor": "#f4f4f9",
                    "padding": "10px",
                    "borderRadius": "5px",
                    "fontSize": "14px",
                    "overflowX": "scroll",
                    "whiteSpace": "pre-wrap",
                },
            ),
            html.H4("Output", style={"marginBottom": "5px"}),
            html.Pre(
                """
Totalt kapital på Avanza: [  1000 139000   2000  30000      0  24000  18000 138000 433000 245000
 222000  57000  83000   7000  19000]
Totalt kapital i Auto: [     0   8000   5000 218000  35000   9000   3000   2000  86000  46000
  11000  14000  48000 108000  18000]
Kapital i Avanza Auto 1: [     0  61000   9000  19000   4000  11000 255000  36000   7000 132000
  10000  35000  27000   2000  80000]
Kapital i Avanza Auto 2: [    0 12000 18000 22000  2000 11000  1000  7000  5000 14000 31000  8000
  3000 20000 42000]
Kapital i Avanza Auto 3: [     0   5000   2000  25000   4000  24000  15000  10000   3000   9000
   1000  40000  65000 207000  14000]
Kapital i Avanza Auto 4: [     0  79000   3000  75000   1000  20000  12000  15000  19000   6000
 150000 309000  24000  73000 119000]
Kapital i Avanza Auto 5: [    0 43000 11000 48000 14000 21000 32000 25000  3000  1000  6000 18000
  5000 59000 30000]
Kapital i Avanza Auto 6: [     0   8000  78000  35000   3000   6000  43000  14000  11000 108000
  24000  10000   1000  54000  13000]
Kapital i aktier: [     0 129000   2000  27000  15000  17000   9000  76000   4000  75000
  31000  49000  14000  36000  43000]
Kapital i fonder (inklusive Auto): [     0   8000   3000   2000 129000 148000 185000  99000  20000   7000
   5000 218000 583000  53000  91000]
Kund sedan år: [2014 2017 2020 2021 2006 2015 2018 2013 2011 2019 2016 2000 2007 2009
 2010]
Kön: ['Företag' 'Man' 'Kvinna']
Åldersintervall: ['-' '31-40' '18-30' '11-17' '71-80' '51-60' '41-50' '61-70' '81-90'
 '0-10' '91+']
Inloggade dagar senaste månaden: [ 1 24  3 11  0 14 13 17 21  5 22 10  2 19  4]
                """,
                style={
                    "backgroundColor": "#f4f4f9",
                    "padding": "10px",
                    "borderRadius": "5px",
                    "fontSize": "14px",
                    "overflowX": "scroll",
                    "whiteSpace": "pre-wrap",
                },
            )
        ],
        style={"textAlign": "left", "lineHeight": "1.6"}
    )
    return create_slide(content, "1. Unika värden i datasetet")

def slide_missing_negative_values():
    content = html.Div(
        children=[
            html.H3("Kod och Dataset: Saknade och negativa värden", style={"marginBottom": "10px", "color": theme["heading_color"]}),
            html.H4("Kodexempel", style={"marginBottom": "5px"}),
            html.Pre(
                """
missing_values = data.isnull().sum()
print("Saknade värden per kolumn:")
print(missing_values)

numerical_columns = data.select_dtypes(include='number').columns
negative_values = data[numerical_columns].lt(0).sum()
print("Negativa värden per kolumn:")
print(negative_values)
                """,
                style={
                    "backgroundColor": "#f4f4f9",
                    "padding": "10px",
                    "borderRadius": "5px",
                    "fontSize": "14px",
                    "overflowX": "scroll",
                    "whiteSpace": "pre-wrap",
                },
            ),
            html.H4("Output", style={"marginBottom": "5px"}),
            html.Pre(
                """
Saknade värden per kolumn:
Totalt kapital på Avanza             0
Totalt kapital i Auto                0
Kapital i Avanza Auto 1              0
Kapital i Avanza Auto 2              0
Kapital i Avanza Auto 3              0
Kapital i Avanza Auto 4              0
Kapital i Avanza Auto 5              0
Kapital i Avanza Auto 6              0
Kapital i aktier                     0
Kapital i fonder (inklusive Auto)    0
Kund sedan år                        0
Kön                                  0
Åldersintervall                      0
Inloggade dagar senaste månaden      0
dtype: int64

Negativa värden per kolumn:
Totalt kapital på Avanza             0
Totalt kapital i Auto                0
Kapital i Avanza Auto 1              0
Kapital i Avanza Auto 2              0
Kapital i Avanza Auto 3              0
Kapital i Avanza Auto 4              0
Kapital i Avanza Auto 5              0
Kapital i Avanza Auto 6              0
Kapital i aktier                     1
Kapital i fonder (inklusive Auto)    0
Kund sedan år                        0
Inloggade dagar senaste månaden      0
dtype: int64
                """,
                style={
                    "backgroundColor": "#f4f4f9",
                    "padding": "10px",
                    "borderRadius": "5px",
                    "fontSize": "14px",
                    "overflowX": "scroll",
                    "whiteSpace": "pre-wrap",
                },
            )
        ],
        style={"textAlign": "left", "lineHeight": "1.6"}
    )
    return create_slide(content, "1. Saknade och negativa värden")

def slide_descriptive_statistics():
    content = html.Div(
        children=[
            html.H3("Kod och Dataset: Deskriptiv statistik och kategoriska frekvenser", style={"marginBottom": "10px", "color": theme["heading_color"]}),
            html.H4("Kodexempel", style={"marginBottom": "5px"}),
            html.Pre(
                """
# Numeriska kolumner
print(data.describe())

# Kategoriska kolumner
categorical_columns = ["Kön", "Åldersintervall"]
for col in categorical_columns:
    print(f"Frekvenser för {col}:")
    print(data[col].value_counts())
                """,
                style={
                    "backgroundColor": "#f4f4f9",
                    "padding": "10px",
                    "borderRadius": "5px",
                    "fontSize": "14px",
                    "overflowX": "scroll",
                    "whiteSpace": "pre-wrap",
                },
            ),
            html.H4("Output", style={"marginBottom": "5px"}),
            html.Pre(
                """
Totalt kapital på Avanza  ...  Inloggade dagar senaste månaden
count              2.000000e+04  ...                     20000.000000
mean               4.868195e+05  ...                         8.948400
std                4.037204e+06  ...                         9.686451
min                0.000000e+00  ...                         0.000000
25%                7.000000e+03  ...                         0.000000
50%                5.200000e+04  ...                         4.000000
75%                2.540000e+05  ...                        18.000000
max                3.011580e+08  ...                        32.000000

[8 rows x 12 columns]
Frekvenser för Kön:
Kön
Man        12365
Kvinna      7269
Företag      366
Name: count, dtype: int64
Frekvenser för Åldersintervall:
Åldersintervall
18-30    5856
31-40    4887
41-50    3302
51-60    2507
61-70    1435
71-80     782
11-17     457
-         366
0-10      256
81-90     136
91+        16
Name: count, dtype: int64
                """,
                style={
                    "backgroundColor": "#f4f4f9",
                    "padding": "10px",
                    "borderRadius": "5px",
                    "fontSize": "14px",
                    "overflowX": "scroll",
                    "whiteSpace": "pre-wrap",
                },
            )
        ],
        style={"textAlign": "left", "lineHeight": "1.6"}
    )
    return create_slide(content, "1. Deskriptiv statistik och kategoriska frekvenser")

def slide_visualizations():
    content = html.Div(
        children=[
            html.H3("Visualiseringar: Korrelationsmatris, scatter plots och boxplots", style={"marginBottom": "20px", "color": theme["heading_color"]}),            
            # Korrelationsmatris med outliers
            dbc.Card(
                [
                    dbc.CardHeader("Korrelationsmatris (med outliers)", style={"textAlign": "center"}),
                    dbc.CardBody(
                        dcc.Graph(
                            id="correlation-matrix",
                            figure=ff.create_annotated_heatmap(
                                z=correlation_matrix.values,
                                x=list(correlation_matrix.columns),
                                y=list(correlation_matrix.index),
                                annotation_text=correlation_matrix.round(2).values,
                                colorscale="Viridis",
                                showscale=True,
                            ),
                        )
                    ),
                ],
                style={"marginBottom": "30px"},
            ),
            
            # Korrelationsmatris utan outliers
            dbc.Card(
                [
                    dbc.CardHeader("Korrelationsmatris (utan outliers)", style={"textAlign": "center"}),
                    dbc.CardBody(
                        dcc.Graph(
                            id="correlation-matrix-no-outliers",
                            figure=ff.create_annotated_heatmap(
                                z=correlation_matrix_no_outliers.values,
                                x=list(correlation_matrix_no_outliers.columns),
                                y=list(correlation_matrix_no_outliers.index),
                                annotation_text=correlation_matrix_no_outliers.round(2).values,
                                colorscale="Viridis",
                                showscale=True,
                            ),
                        )
                    ),
                ],
                style={"marginBottom": "30px"},
            ),

            # Scatter plots
            dbc.Row(
                [
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader("Samband: Inloggade dagar och totalt kapital (med outliers)", style={"textAlign": "center"}),
                                dbc.CardBody(
                                    dcc.Graph(
                                        id="scatter-login-vs-capital",
                                        figure=px.scatter(
                                            data,
                                            x="Inloggade dagar senaste månaden",
                                            y="Totalt kapital på Avanza",
                                            title="Samband mellan inloggade dagar och kapital (med outliers)",
                                            labels={"x": "Inloggade dagar senaste månaden", "y": "Totalt kapital på Avanza"},
                                            template="plotly",
                                        ),
                                    )
                                ),
                            ]
                        ),
                        width=6,
                    ),
                    dbc.Col(
                        dbc.Card(
                            [
                                dbc.CardHeader("Samband: Inloggade dagar och totalt kapital (utan outliers)", style={"textAlign": "center"}),
                                dbc.CardBody(
                                    dcc.Graph(
                                        id="scatter-login-vs-capital-no-outliers",
                                        figure=px.scatter(
                                            data_no_outliers,
                                            x="Inloggade dagar senaste månaden",
                                            y="Totalt kapital på Avanza",
                                            title="Samband mellan inloggade dagar och kapital (utan outliers)",
                                            labels={"x": "Inloggade dagar senaste månaden", "y": "Totalt kapital på Avanza"},
                                            template="plotly",
                                        ),
                                    )
                                ),
                            ]
                        ),
                        width=6,
                    ),
                ],
                style={"marginBottom": "30px"},
            ),

            # Boxplots
            dbc.Card(
                [
                    dbc.CardHeader("Boxplot för numeriska variabler fördelade på kategoriska kolumner (med och utan outliers)", style={"textAlign": "center"}),
                    dbc.CardBody(
                        html.Div(
                            [
                                # First four boxplots
                                *[
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                dcc.Graph(
                                                    id=f"boxplot-with-outliers-{num_col}-{cat_col}",
                                                    figure=px.box(
                                                        data,
                                                        x=cat_col,
                                                        y=num_col,
                                                        title=f"Boxplot av {num_col} fördelat på {cat_col} (med outliers)",
                                                        labels={cat_col: cat_col, num_col: num_col},
                                                        template="plotly",
                                                    ),
                                                ),
                                                width=6,
                                            ),
                                            dbc.Col(
                                                dcc.Graph(
                                                    id=f"boxplot-without-outliers-{num_col}-{cat_col}",
                                                    figure=px.box(
                                                        data_no_outliers,
                                                        x=cat_col,
                                                        y=num_col,
                                                        title=f"Boxplot av {num_col} fördelat på {cat_col} (utan outliers)",
                                                        labels={cat_col: cat_col, num_col: num_col},
                                                        template="plotly",
                                                    ),
                                                ),
                                                width=6,
                                            ),
                                        ]
                                    )
                                    for num_col, cat_col in zip(numerical_columns[:4], categorical_columns[:4])
                                ],

                                # Ellipsis to indicate skipped content
                                html.Div(
                                    "•••", 
                                    style={
                                        "textAlign": "center", 
                                        "fontSize": "24px", 
                                        "margin": "20px 0"
                                    }
                                ),

                                # Last four boxplots
                                *[
                                    dbc.Row(
                                        [
                                            dbc.Col(
                                                dcc.Graph(
                                                    id=f"boxplot-with-outliers-{num_col}-{cat_col}",
                                                    figure=px.box(
                                                        data,
                                                        x=cat_col,
                                                        y=num_col,
                                                        title=f"Boxplot av {num_col} fördelat på {cat_col} (med outliers)",
                                                        labels={cat_col: cat_col, num_col: num_col},
                                                        template="plotly",
                                                    ),
                                                ),
                                                width=6,
                                            ),
                                            dbc.Col(
                                                dcc.Graph(
                                                    id=f"boxplot-without-outliers-{num_col}-{cat_col}",
                                                    figure=px.box(
                                                        data_no_outliers,
                                                        x=cat_col,
                                                        y=num_col,
                                                        title=f"Boxplot av {num_col} fördelat på {cat_col} (utan outliers)",
                                                        labels={cat_col: cat_col, num_col: num_col},
                                                        template="plotly",
                                                    ),
                                                ),
                                                width=6,
                                            ),
                                        ]
                                    )
                                    for num_col, cat_col in zip(numerical_columns[-4:], categorical_columns[-4:])
                                ],
                            ]
                        )
                    ),
                ],
                style={"marginBottom": "30px"},
            ),
        ],
        style={"textAlign": "left", "lineHeight": "1.6"},
    )
    return create_slide(content, "1. Utforskande visualiseringar")


def slide_gender_age_distribution():
    content = html.Div(
        children=[
            # Könsfördelning
            dbc.Card(
                [
                    dbc.CardHeader("Majoriteten av kunderna är män (cirka 61.8%), följt av kvinnor (36.3%) och företag (1.9%).", style={"textAlign": "center"}),
                    dbc.CardBody(
                        dcc.Graph(
                            id="gender-distribution",
                            figure=px.bar(
                                gender_distribution,
                                title="Könsfördelning",
                                labels={"index": "Kön", "value": "Antal kunder"},
                                color_discrete_sequence=["#636EFA"],
                            ),
                        )
                    ),
                ],
                style={"marginBottom": "30px"},
            ),
            
            # Åldersfördelning
            dbc.Card(
                [
                    dbc.CardHeader("Den största åldersgruppen är 18–30 år (29.3%), medan mycket få kunder är över 81 år.", style={"textAlign": "center"}),
                    dbc.CardBody(
                        dcc.Graph(
                            id="age-distribution",
                            figure=px.bar(
                                age_distribution,
                                title="Åldersfördelning",
                                labels={"index": "Åldersintervall", "value": "Antal kunder"},
                                color_discrete_sequence=["#EF553B"],
                            ),
                        )
                    ),
                ],
                style={"marginBottom": "30px"},
            ),
        ],
        style={"textAlign": "left", "lineHeight": "1.6"},
    )
    return create_slide(content, "2. Köns- och åldersfördelning")

def slide_activity_and_auto_analysis():
    content = html.Div(
        children=[
          
            # Inloggningar per kön och åldersgrupp
            dbc.Card(
                [
                    dbc.CardHeader("Företagskunder har i genomsnitt 6.47 inloggningar per månad. Män i åldersgruppen 31–40 år är mest aktiva, med över 11 inloggningar per månad.)", style={"textAlign": "center"}),
                    dbc.CardBody(
                        dcc.Graph(
                            id="login-activity",
                            figure=px.bar(
                                login_activity,
                                title="Genomsnittligt antal inloggningar per månad (kön & ålder)",
                                labels={"value": "Inloggningar", "index": "Kön"},
                                barmode="stack",
                                color_discrete_sequence=px.colors.qualitative.Pastel,
                            ),
                        )
                    ),
                ],
                style={"marginBottom": "30px"},
            ),

            # Kapital och aktivitet (per kön)
            dbc.Card(
                [
                    dbc.CardHeader("Företagskunder har det högsta genomsnittliga totalkapitalet (cirka 4.17 miljoner SEK) men lägre aktivitet (6.47 inloggningar per månad). Män har betydligt högre genomsnittligt kapital (ca 507,000 SEK) än kvinnor (ca 266,000 SEK) och är mer aktiva (10.66 inloggningar per månad jämfört med kvinnors 6.16).", style={"textAlign": "center"}),
                    dbc.CardBody(
                        dcc.Graph(
                            id="activity-vs-capital",
                            figure=px.bar(
                                activity_vs_capital,
                                title="Genomsnittligt kapital och aktivitet (per kön)",
                                labels={"value": "Genomsnitt", "index": "Kön"},
                                color_discrete_sequence=["#00CC96"],
                            ),
                        )
                    ),
                ],
                style={"marginBottom": "30px"},
            ),

            # Auto-kapital per åldersgrupp
            dbc.Card(
                [
                    dbc.CardHeader("Den största Auto-investeringen sker i åldersgruppen 31–40 år, följt av 41–50 år. Äldre åldersgrupper (61+ år) tenderar att ha mindre kapital investerat i Auto-produkter.", style={"textAlign": "center"}),
                    dbc.CardBody(
                        dcc.Graph(
                            id="auto-investments",
                            figure=px.bar(
                                auto_investments,
                                title="Totalt Auto-kapital per åldersgrupp",
                                labels={"value": "Kapital i Auto-produkter", "index": "Åldersintervall"},
                                barmode="stack",
                                color_discrete_sequence=px.colors.qualitative.Safe,
                            ),
                        )
                    ),
                ],
                style={"marginBottom": "30px"},
            ),

            # Auto-fördelning per kön och åldersgrupp
            dbc.Card(
                [
                    dbc.CardHeader("Kvinnor investerar i genomsnitt en något större andel av sitt kapital i Auto-produkter jämfört med män i de flesta åldersgrupper. Den högsta Auto-andelen syns bland kvinnor i åldern 31–40 år (6.6%) och män i samma grupp (3.9%).", style={"textAlign": "center"}),
                    dbc.CardBody(
                        dcc.Graph(
                            id="auto-preferences",
                            figure=px.bar(
                                auto_preferences,
                                title="Andel av kapital i Auto-produkter (per kön och åldersgrupp)",
                                labels={"value": "Auto-andel av totalkapital (%)", "index": "Kön"},
                                barmode="group",
                                color_discrete_sequence=px.colors.qualitative.Vivid,
                            ),
                        )
                    ),
                ],
                style={"marginBottom": "30px"},
            ),

            # Korrelation mellan aktivitet och Auto-investeringar
            dbc.Card(
                [
                    dbc.CardHeader("Sambandet mellan antalet inloggningar och Auto-fördelning är relativt svagt men indikerar att mer aktiva användare inte nödvändigtvis har större Auto-andel i sitt kapital.", style={"textAlign": "center"}),
                    dbc.CardBody(
                        dcc.Graph(
                            id="activity-auto-correlation",
                            figure=px.scatter(
                                data,
                                x="Inloggade dagar senaste månaden",
                                y="Auto vs Non-Auto",
                                title="Korrelation: Inloggningar och Auto-fördelning",
                                labels={"x": "Inloggade dagar senaste månaden", "y": "Auto vs Non-Auto andel"},
                                color="Kön",
                                color_discrete_sequence=px.colors.qualitative.Dark24,
                                opacity=0.7,
                            ),
                        )
                    ),
                ],
                style={"marginBottom": "30px"},
            ),
        ],
        style={"textAlign": "left", "lineHeight": "1.6"},
    )
    return create_slide(content, "2. Aktivitet och Auto-investeringar")

def slide_further_investigation():
    content = html.Div(
        children=[
            html.H3("3. Vad vi borde gräva vidare i", style={"marginBottom": "20px", "color": theme["heading_color"]}),
            
            # Kundsegmentering
            html.H4("1. Kundsegmentering", style={"marginBottom": "10px"}),
            html.Ul(
                [
                    html.Li("Bred kundbas, behöver brytas ned för att förstå varje segment/grupp av kunder."),
                    html.Li("Identifiera kundgrupper baserat på kapitalfördelning, ålder, kön, och aktivitet."),
                ],
                style={"marginBottom": "20px"}
            ),

            # Kundresan och förändring över tid
            html.H4("2. Kundresan och förändring över tid", style={"marginBottom": "10px"}),
            html.Ul(
                [
                    html.Li("Andelen potentiella kunder (t.ex. de som besöker Avanza Auto-sidan) som faktiskt investerar i produkten."),
                    html.Li("Identifiera vilka klick som ökar sannolikheten att bli kund?"),
                    html.Li("Hur har deras investeringsmönster förändrats sedan de blev kunder?"),
                ],
                style={"marginBottom": "20px"}
            ),

            # Datasetets begränsningar
            html.H4("3. Datasetet har vissa begränsningar", style={"marginBottom": "10px"}),
            html.Ul(
                [
                    html.Li("Saknade datapunkter:"),
                    html.Ul(
                        [
                            html.Li("Tidsdimension: Data är 'statisk' och fångar inte utvecklingen över tid för kunder."),
                            html.Li("Flöden: Insättningar och uttag över tid."),
                            html.Li("Inkomstnivå eller ekonomisk status: Kan ge kontext till deras investeringskapacitet."),
                            html.Li("Utbildningsnivå eller ekonomisk kunskap: Hjälper att förstå deras preferenser för självstyrda investeringar vs. automatiserade lösningar."),
                            html.Li("Geografisk fördelning: Kan visa regionala skillnader i investeringsbeteenden."),
                            html.Li("Engagemang utanför inloggningar: Exempelvis deltagande i seminarier eller kontakt med rådgivare."),
                            html.Li("Livshändelser: Större ekonomiska händelser som kan påverka sparande eller investeringsmönster."),
                        ]
                    ),
                ],
                style={"marginBottom": "20px"}
            ),

            # Relevant avancerad modellering och implementering
            html.H4("Relevant avancerad modellering och implementering", style={"marginBottom": "10px"}),
            html.Ul(
                [
                    html.Li("Kundsegmentering med klustring (Clustering):"),
                    html.Ul(
                        [
                            html.Li("Använd en algoritm som K-means eller DBSCAN för att identifiera kundsegment."),
                            html.Li("Egenskaper som kan användas:"),
                            html.Ul(
                                [
                                    html.Li("Kapital i olika kategorier (Auto, aktier, fonder)."),
                                    html.Li("Aktivitet (antal inloggningar)."),
                                    html.Li("Ålder och kön."),
                                ]
                            ),
                        ]
                    ),
                ]
            ),
        ],
        style={"textAlign": "left", "lineHeight": "1.6"},
    )
    return create_slide(content, "3. Vidare undersökning")

def slide_clustering_analysis():
    content = html.Div(
        children=[
            html.H3("Klustringsanalys: Visualiseringar", style={"marginBottom": "20px", "color": theme["heading_color"]}),
            
            # Elbow Method Graph
            dbc.Card(
                [
                    dbc.CardHeader("Optimal antal kluster med Elbow-metoden", style={"textAlign": "center"}),
                    dbc.CardBody(
                        html.Iframe(
                            src="/assets/elbow_method.html",  # Assuming the file is in the 'assets' folder
                            style={
                                "border": "none",
                                "width": "100%",
                                "height": "600px"
                            }
                        )
                    ),
                ],
                style={"marginBottom": "30px"},
            ),

            # Silhouette Score Graph
            dbc.Card(
                [
                    dbc.CardHeader("Silhouette Score för Klustring", style={"textAlign": "center"}),
                    dbc.CardBody(
                        html.Iframe(
                            src="/assets/silhouette_score.html",  # Assuming the file is in the 'assets' folder
                            style={
                                "border": "none",
                                "width": "100%",
                                "height": "600px"
                            }
                        )
                    ),
                ],
                style={"marginBottom": "30px"},
            ),
        ],
        style={"textAlign": "left", "lineHeight": "1.6"},
    )
    return create_slide(content, "3. Klustringsanalys: Visualiseringar")

def slide_clustering_visualizations():
    content = html.Div(
        children=[
            html.H3("Kundsegmentering: Visualiseringar med K-means", style={"marginBottom": "20px", "color": theme["heading_color"]}),
            
            # Embed the pre-generated HTML file
            dbc.Card(
                [
                    dbc.CardHeader("Kundsegmentering med K-means (förgenererad visualisering)", style={"textAlign": "center"}),
                    dbc.CardBody(
                        html.Iframe(
                            src="/assets/cluster_visualization.html",  # Assuming the file is in the 'assets' folder
                            style={
                                "border": "none",
                                "width": "100%",
                                "height": "600px"
                            }
                        )
                    ),
                ],
                style={"marginBottom": "30px"},
            ),
        ],
        style={"textAlign": "left", "lineHeight": "1.6"},
    )
    return create_slide(content, "3. Kundsegmentering: Visualiseringar")



def slide_thank_you():
    content = html.Div(
        children=[
            html.H3("Tack för er tid!", style={"marginBottom": "20px", "color": theme["heading_color"], "textAlign": "center"}),
            html.P(
                "Om ni har några frågor eller funderingar, tveka inte att höra av er!",
                style={"marginBottom": "20px", "textAlign": "center", "fontSize": "18px"}
            ),
            html.Div(
                children=[
                    html.P("Farzad Ashouri", style={"fontWeight": "bold", "textAlign": "center", "fontSize": "18px"}),
                    html.P("E-post: ashouri.farzad@gmail.com", style={"textAlign": "center", "fontSize": "16px"}),
                    html.P("Telefon: +46 (0) 70-742 23 39", style={"textAlign": "center", "fontSize": "16px"}),
                ],
                style={"marginBottom": "30px"}
            ),
            html.Div(
                "Lycka till med era analyser och beslut!", 
                style={"textAlign": "center", "fontStyle": "italic", "fontSize": "16px"}
            )
        ],
        style={
            "backgroundColor": theme["background_content"],
            "padding": "40px",
            "borderRadius": "10px",
            "boxShadow": "0 4px 6px rgba(0, 0, 0, 0.1)",
            "textAlign": "center",
            "lineHeight": "1.6",
        },
    )
    return create_slide(content, "Tack")



slides = [
            slide_1, slide_2, slide_kpi, slide_recommendation, 
            slide_challenges, slide_ml_models, 
            slide_churn_example, slide_customer_analysis,
            slide_dataset_analysis, slide_unique_values, slide_missing_negative_values,
            slide_descriptive_statistics, slide_visualizations, 
            slide_gender_age_distribution, slide_activity_and_auto_analysis,
            slide_further_investigation, slide_clustering_analysis,
            slide_clustering_visualizations, slide_thank_you
            
        ]

app.layout = html.Div(
    style={
        "backgroundColor": theme["background_page"],
        "padding": "20px",
        "minHeight": "100vh",
        "fontFamily": theme["font_family"],
    },
    children=[
        dcc.Store(id="slide-index", data=0),  # To store the current slide index

        # Navigation buttons at the top
        html.Div(
            style={
                "display": "flex",
                "justifyContent": "space-between",
                "marginBottom": "30px",  # Space below the top buttons
            },
            children=[
                html.Button(
                    "Föregående",
                    id="prev-slide-top",
                    n_clicks=0,
                    style={
                        "backgroundColor": theme["primary_color"],
                        "color": "#ffffff",
                        "border": "none",
                        "padding": "10px 20px",
                        "borderRadius": "5px",
                        "cursor": "pointer",
                    },
                ),
                html.Button(
                    "Nästa",
                    id="next-slide-top",
                    n_clicks=0,
                    style={
                        "backgroundColor": theme["primary_color"],
                        "color": "#ffffff",
                        "border": "none",
                        "padding": "10px 20px",
                        "borderRadius": "5px",
                        "cursor": "pointer",
                    },
                ),
            ],
        ),

        # Slide container
        html.Div(
            id="slide-container",
            style={"marginTop": "20px"},
            children=[slides[0]()],
        ),

        # Navigation buttons at the bottom
        html.Div(
            style={
                "display": "flex",
                "justifyContent": "space-between",
                "marginTop": "30px",  # Space above the bottom buttons
            },
            children=[
                html.Button(
                    "Föregående",
                    id="prev-slide-bottom",
                    n_clicks=0,
                    style={
                        "backgroundColor": theme["primary_color"],
                        "color": "#ffffff",
                        "border": "none",
                        "padding": "10px 20px",
                        "borderRadius": "5px",
                        "cursor": "pointer",
                    },
                ),
                html.Button(
                    "Nästa",
                    id="next-slide-bottom",
                    n_clicks=0,
                    style={
                        "backgroundColor": theme["primary_color"],
                        "color": "#ffffff",
                        "border": "none",
                        "padding": "10px 20px",
                        "borderRadius": "5px",
                        "cursor": "pointer",
                    },
                ),
            ],
        ),
    ],
)

@callback(
    [Output("slide-container", "children"), Output("slide-index", "data")],
    [
        Input("prev-slide-top", "n_clicks"),
        Input("next-slide-top", "n_clicks"),
        Input("prev-slide-bottom", "n_clicks"),
        Input("next-slide-bottom", "n_clicks"),
    ],
    State("slide-index", "data"),
)
def update_slide(prev_top, next_top, prev_bottom, next_bottom, slide_index):
    ctx = dash.callback_context
    if not ctx.triggered:
        return [slides[slide_index](), slide_index]

    # Determine which button was clicked
    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
    
    # Update slide index based on the button clicked
    if triggered_id in ["next-slide-top", "next-slide-bottom"]:
        slide_index = (slide_index + 1) % len(slides)
    elif triggered_id in ["prev-slide-top", "prev-slide-bottom"]:
        slide_index = (slide_index - 1) % len(slides)

    # Return updated slide and index
    return [slides[slide_index](), slide_index]






if __name__ == "__main__":
    app.run_server(debug=False)
