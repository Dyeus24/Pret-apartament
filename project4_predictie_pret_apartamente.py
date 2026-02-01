# Predictie Pret Apartamente
import pandas as pd
from sklearn.linear_model import LinearRegression

def main():
    df = pd.DataFrame({
        'mp': [50, 60, 70],
        'pret': [60000, 72000, 85000]
    })
    model = LinearRegression()
    model.fit(df[['mp']], df['pret'])
    print("Coef:", model.coef_)

if __name__ == "__main__":
    main()
