using Microsoft.VisualBasic.FileIO;
using Microsoft.VisualStudio.TestTools.UnitTesting;
using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.util;
using XGBoost;


namespace XGBoost_first
{

    class Program
    {
        public const int iloscKlas = 3; // Definiujemy ile mamy miec klas na wyjsciu
        public const int iloscParametrow = 4; // Definiujemy ile jest parametrow na podstawie ktorych bedziemy decydowali o przynaleznosci do klasy


        static void Main(string[] args)
        {
            // Inicjalizacja klasyfikatora XGBoost. 
            var xgb = new XGBClassifier(objective: "multi:softprob", numClass: iloscKlas);

            // Inicjalizacja list potrzebnych do przekopiowania csv do pamieci programu
            List<float[]> records = new List<float[]>();
            List<float> labels = new List<float>();
            float[][] records_array;
            float[] label_array;

            // Rozpoczecie czytania z pliku
            using (var reader = new StreamReader(@"C:\Users\Michal\source\repos\XGBoost_first\iris.csv"))
            {
                
                while (!reader.EndOfStream)
                {
                    var line = reader.ReadLine();
                    var values = line.Split(',');
                    float[] wiersz = new float[iloscParametrow];
                    wiersz[0] = float.Parse(values[0], CultureInfo.InvariantCulture.NumberFormat);
                    wiersz[1] = float.Parse(values[1], CultureInfo.InvariantCulture.NumberFormat);
                    wiersz[2] = float.Parse(values[2], CultureInfo.InvariantCulture.NumberFormat);
                    wiersz[3] = float.Parse(values[3], CultureInfo.InvariantCulture.NumberFormat);
                    string Label = values[4];

                    // Ponizsze warunki sa tylko po to zeby przekonwertowac wartosci tekstowe etykiet na liczbowe
                    if (Label.Contains("setosa"))
                    {
                        labels.Add(0.0f);
                    }
                    else if (Label.Contains("versicolor"))
                    {
                        labels.Add(1.0f);
                    }
                    else if (Label.Contains("virginica"))
                    {
                        labels.Add(2.0f);
                    }
                    records.Add(wiersz);
                }
               // Konwersja list do tablicy, tak aby moc wywolac funkcje fit
                records_array = records.ToArray();
                label_array = labels.ToArray();
            }

            // Sprawdzamy czy ilosc rekordow jest rowna ilosci etykiet
            Assert.AreEqual(records_array.Length, label_array.Length);


           // Parametry algorytmu ustawiamy poleceniem setparameter, jako argument podajac nazwe parametru i jego wartosc
           // Te parametry mozna takze podac podczas inicjalizacji klasyfikatora
           xgb.SetParameter("max_depth", 3);
            /* LISTA PARAMETRÓW:
        ["max_depth"] = 3,
        ["learning_rate"] = 0.1f,
        ["n_estimators"] = 100,
        ["silent"] = true,
        ["objective"] = "binary:logistic",
        ["booster"] = "gbtree",
        ["tree_method"] = "auto",
        ["nthread"] = -1,
        ["gamma"] = 0,
        ["min_child_weight"] = 1,
        ["max_delta_step"] = 0,
        ["subsample"] = 1,
        ["colsample_bytree"] = 1,
        ["colsample_bylevel"] = 1,
        ["reg_alpha"] = 0,
        ["reg_lambda"] = 1,
        ["scale_pos_weight"] = 1,
        ["sample_type"] = "uniform",
        ["normalize_type"] = "tree",
        ["rate_drop"] = 0.0f,
        ["one_drop"] = 0,
        ["skip_drop"] = 0f,
        ["base_score"] = 0.5f,
        ["seed"] = 0,
        ["missing"] = float.NaN,
        ["_Booster"] = null,
        ["num_class"] = 0
             */
            // Trenujemy drzewo decyzyjne!
            xgb.Fit(records_array, label_array);



            // Wektor do testow
            float[][] vectorsTest = new float[][]
            {
            new[] { 5.8f, 2.7f, 5.1f, 1.9f},
            new[] { 5.0f, 3.4f, 1.6f ,0.4f},
            };


            // Ponizej funkcje do predyckji. Pierwsza zwraca dokladny wynik prawdopodobienstwa przynaleznosci do danej klasy
            // Druga zwraca wartosci zerojedynkowe. Mozna sobie potestowac
            float[] labelsTestPredicted = xgb.PredictRaw(vectorsTest);
            //float[] labelsTestPredicted = xgb.Predict(vectorsTest);
     
            // Wypisywanie przewidzianych wartosci
            for (int i = 0;i< labelsTestPredicted.Length;i++)
                Console.WriteLine(labelsTestPredicted[i]);

        
    }
    }
}
