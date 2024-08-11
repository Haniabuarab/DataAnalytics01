{
  "nbformat": 4,
  "nbformat_minor": 0,
  "metadata": {
    "colab": {
      "provenance": [],
      "mount_file_id": "1-iq5Wp2l3KlKPjEUEhT2ajLE9_DpcT1H",
      "authorship_tag": "ABX9TyMUVTjDg83+2fEL//AhZcew"
    },
    "kernelspec": {
      "name": "python3",
      "display_name": "Python 3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "cells": [
    {
      "cell_type": "code",
      "source": [
        "# prompt: write knn model to classify certain data\n",
        "\n",
        "from sklearn.model_selection import train_test_split\n",
        "from sklearn.neighbors import KNeighborsClassifier\n",
        "from sklearn.preprocessing import StandardScaler\n",
        "\n",
        "# Assuming 'target_variable' is the name of the column you want to predict\n",
        "X = fifa_data.drop('target_variable', axis=1)  # Replace 'target_variable'\n",
        "y = fifa_data['target_variable']\n",
        "\n",
        "# Split the data into training and testing sets\n",
        "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n",
        "\n",
        "# Scale the features using StandardScaler\n",
        "scaler = StandardScaler()\n",
        "X_train = scaler.fit_transform(X_train)\n",
        "X_test = scaler.transform(X_test)\n",
        "\n",
        "# Create a KNN classifier object\n",
        "knn = KNeighborsClassifier(n_neighbors=5)  # You can adjust the number of neighbors\n",
        "\n",
        "# Train the model\n",
        "knn.fit(X_train, y_train)\n",
        "\n",
        "# Make predictions on the test set\n",
        "y_pred = knn.predict(X_test)\n",
        "\n",
        "# Evaluate the model\n",
        "from sklearn.metrics import accuracy_score, classification_report\n",
        "print(\"Accuracy:\", accuracy_score(y_test, y_pred))\n",
        "print(\"\\nClassification Report:\\n\", classification_report(y_test, y_pred))\n"
      ],
      "metadata": {
        "id": "xHQMyxVr8A5G"
      },
      "execution_count": null,
      "outputs": []
    }
  ]
}