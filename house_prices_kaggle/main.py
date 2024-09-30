import dill
import pandas as pd

with open('model/house_prices_regressor.pkl', 'rb') as f:
    model = dill.load(f)

test_df = pd.read_csv('../data/test.csv')

predictions = model['model'].predict(test_df)

prediction_df = pd.DataFrame({
    'Id': test_df['Id'],
    'SalePrice': predictions
})

output_file_path = 'model/house_prices_prediction.csv'
prediction_df.to_csv(output_file_path, index=False)

print(f'Predictions saved to {output_file_path}')
