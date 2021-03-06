DATASETS_URL = "https://www.kaggle.com/murderaccountability/homicide-reports/downloads/homicide-reports.zip"

import os
import zipfile


MURDER_PATH = "datasets/murder"

zip_path = os.path.join(MURDER_PATH, "homicide-reports.zip")
murder_zip = zipfile.ZipFile(zip_path)
murder_zip.extractall(path=MURDER_PATH)
murder_zip.close()

