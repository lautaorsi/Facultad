import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


matrix_EN = np.array([[288,212],[233,267]])
matrix_HH = np.array([[340,160],[241,259]])
matrix_GS = np.array([[66,434],[125,375]])
matrix_SVD = np.array([[0,0],[0,0]])  # Ajusta esto con tus datos reales

# Función para calcular métricas de una matriz de confusión
def calcular_metricas(conf_matrix):
    TP = conf_matrix[1, 1]  # Verdaderos Positivos (perros correctos)
    TN = conf_matrix[0, 0]  # Verdaderos Negativos (gatos correctos)
    FP = conf_matrix[0, 1]  # Falsos Positivos (gatos predichos como perros)
    FN = conf_matrix[1, 0]  # Falsos Negativos (perros predichos como gatos)
    
    accuracy = (TP + TN) / np.sum(conf_matrix) * 100  # YA EN PORCENTAJE
    precision_perros = TP / (TP + FP) if (TP + FP) > 0 else 0
    precision_gatos = TN / (TN + FN) if (TN + FN) > 0 else 0
    recall_perros = TP / (TP + FN) if (TP + FN) > 0 else 0
    recall_gatos = TN / (TN + FP) if (TN + FP) > 0 else 0
    f1_score_gatos = 2 * (precision_gatos * recall_gatos) / (precision_gatos + recall_gatos) if (precision_gatos + recall_gatos) > 0 else 0
    f1_score_perros = 2 * (precision_perros * recall_perros) / (precision_perros + recall_perros) if (precision_perros + recall_perros) > 0 else 0
    
    return {
        'Accuracy': accuracy,
        'Precision_perros': precision_perros,
        'Precision_gatos': precision_gatos,
        'Recall_perros': recall_perros,
        'Recall_gatos': recall_gatos,
        'F1-Score_perros': f1_score_perros,
        'F1-Score_gatos': f1_score_gatos
    }

# Calcular métricas para cada método
metodos = ['Ec. Normales', 'SVD', 'QR HouseHolder', 'QR Gram-Schmidt']
matrices = [matrix_EN, matrix_SVD, matrix_HH, matrix_GS]

resultados = {}
for metodo, matriz in zip(metodos, matrices):
    resultados[metodo] = calcular_metricas(matriz)

df_resultados = pd.DataFrame(resultados).T

# ===== GRÁFICO 1: ACCURACY =====
fig1, ax1 = plt.subplots(figsize=(10, 6))

x = np.arange(len(metodos))
width = 0.6

accuracy_valores = df_resultados['Accuracy'].values
bars = ax1.bar(x, accuracy_valores, width, color='#3498db', alpha=0.8, edgecolor='black', linewidth=1.5)

ax1.set_xlabel('Modelo', fontsize=13, fontweight='bold')
ax1.set_ylabel('Accuracy (%)', fontsize=13, fontweight='bold')  # CAMBIO: agregado (%)
ax1.set_title('Comparación de Accuracy entre modelos', 
             fontsize=15, fontweight='bold', pad=20)
ax1.set_xticks(x)
ax1.set_xticklabels(metodos, rotation=15, ha='right', fontsize=11)
ax1.set_ylim([0, 105])  # CAMBIO: de 1.05 a 105
ax1.grid(axis='y', alpha=0.3, linestyle='--')

# Agregar valores sobre las barras
for i, (bar, valor) in enumerate(zip(bars, accuracy_valores)):
    height = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2., height + 1.5,  # CAMBIO: +1.5 en vez de +0.02
            f'{valor:.2f}%',  # CAMBIO: agregado % y cambiado a 2 decimales
            ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.show()

# ===== GRÁFICO 2: PRECISION Y RECALL POR CLASE =====
fig2, ax2 = plt.subplots(figsize=(14, 8))

x = np.arange(len(metodos))
width = 0.2

metricas_detalladas = ['Precision_perros', 'Precision_gatos', 'Recall_perros', 'Recall_gatos']
colores = ['#e74c3c', '#c0392b', '#2ecc71', '#27ae60']
labels = ['Precision Perros', 'Precision Gatos', 'Recall Perros', 'Recall Gatos']

for i, (metrica, color, label) in enumerate(zip(metricas_detalladas, colores, labels)):
    valores = df_resultados[metrica].values
    ax2.bar(x + i*width - width*1.5, valores, width, label=label, color=color, alpha=0.85, edgecolor='black', linewidth=0.8)

ax2.set_xlabel('Modelo', fontsize=13, fontweight='bold')
ax2.set_ylabel('Puntaje', fontsize=13, fontweight='bold')
ax2.set_title('Comparación de Precision y Recall', 
             fontsize=15, fontweight='bold', pad=20)
ax2.set_xticks(x)
ax2.set_xticklabels(metodos, rotation=15, ha='right', fontsize=11)
ax2.legend(loc='upper right', fontsize=11, framealpha=0.9)
ax2.set_ylim([0, 1.05])
ax2.grid(axis='y', alpha=0.3, linestyle='--')

# Agregar valores sobre las barras
for i, metrica in enumerate(metricas_detalladas):
    valores = df_resultados[metrica].values
    for j, valor in enumerate(valores):
        ax2.text(j + i*width - width*1.5, valor + 0.02, f'{valor:.2f}', 
                ha='center', va='bottom', fontsize=8, fontweight='bold', rotation=0)

plt.tight_layout()
plt.show()

fig, axes = plt.subplots(2, 2, figsize=(14, 12))
axes = axes.flatten()

for idx, (matriz, titulo) in enumerate(zip(matrices, metodos)):
    sns.heatmap(matriz, annot=True, fmt='g', cmap='Blues', 
                xticklabels=['Gatos', 'Perros'],
                yticklabels=['Gatos', 'Perros'],
                ax=axes[idx],
                cbar_kws={'shrink': 0.8},
                annot_kws={'size': 16, 'weight': 'bold'})  # Tamaño de números
    
    axes[idx].set_ylabel('Etiqueta Real', fontsize=13, fontweight='bold')
    axes[idx].set_xlabel('Predicción', fontsize=13, fontweight='bold')
    axes[idx].set_title(f'Matriz de Confusión - {titulo}', 
                       fontsize=14, fontweight='bold', pad=15)
    
    # Ajustar tamaño de las etiquetas de los ejes
    axes[idx].tick_params(axis='both', which='major', labelsize=12)

plt.tight_layout()
plt.show()

# ===== TABLA DE RESULTADOS =====
print("\n" + "="*90)
print("TABLA COMPARATIVA DE RESULTADOS")
print("="*90)
print(df_resultados.round(4).to_string())
print("="*90)

# ===== ANÁLISIS ADICIONAL =====
print("\n" + "="*90)
print("ANÁLISIS DE RESULTADOS")
print("="*90)
for metodo in metodos:
    print(f"\n{metodo}:")
    print(f"  Accuracy:         {df_resultados.loc[metodo, 'Accuracy']:.2f}%")  # CAMBIO: agregado % y 2 decimales
    print(f"  Precision Perros: {df_resultados.loc[metodo, 'Precision_perros']:.4f}")
    print(f"  Precision Gatos:  {df_resultados.loc[metodo, 'Precision_gatos']:.4f}")
    print(f"  Recall Perros:    {df_resultados.loc[metodo, 'Recall_perros']:.4f}")
    print(f"  Recall Gatos:     {df_resultados.loc[metodo, 'Recall_gatos']:.4f}")
print("="*90)