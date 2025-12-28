# Ferramenta de ensaio de estabilidade para MLPs
> Ferramenta experimental para análise de estabilidade, convergência e impacto de seeds em MLPs configuradas via YAML.

Este projeto implementa um script para ensaio de estabilidade em redes neurais do tipo MLP, configuradas por meio de arquivos YAML, o qual permite configurar as camadas, funções de ativação, *seeds*, normalização, otimizador, função de custo e hiperparâmetros gerais. O script funciona com base de dados em sqlite ou CSV.

O foco é verificar como o impacto de diferentes seeds influencia na convergência de um determinado modelo de MLP, testando desta forma suas capacidades de generalização e de contornar mínimos locais. Adicionalmente, também permite verificar o impacto de diferentes normalizações (*minmax* e *standard*) no treinamento do modelo.

## 🔍 Visão Geral

O script realiza uma série de treinamentos a partir de configurações definidos em um arquivo YAML fornecido como parâmetro de entrada, coletando métricas de desempenho (R2 para regressão e acurácia para classificação) e evolução temporal da função custo. 

Atualmente, há dois tipos de experimentos possíveis: verificação do impacto de normalização e verificação da estabilidade a partir de diferentes seeds. Os experimentos geram arquivos de relatório em CSV contendo todas as rodadas de treinamento, um arquivo de sumário geral e gráficos para a função custo de cada treinamento. 

O projeto foi desenvolvido com o intuito de criar uma ferramenta que, ao mesmo tempo, permita extrair estatísticas de treinamento para análise e automatizar processos de verificação de performance para diferentes descrições de modelos.   

## ▶️ Como executar 
### 1. Clone o repositório
```bash
git clone https://github.com/pedro123k/mlp-lab.git  
cd mlp-lab
``` 
### 2. Crie um ambiente virtual
```bash
python -m venv ./venv
source .venv/bin/activate # Linux / Mac
# ou
.\.venv\Scripts\activate #Windows
```

### 3. Instale as dependências 
```bash
pip install -r requirements.txt
```

### 4. Execute o script 
```bash
python lab.py --config=configs/teste1.yaml 
```

### Parâmetros adicionais
```bash
--outdir # Define um diretório de saída diferente do padrão (results)
--label # Define um identificador para o nome dos arquivos gerados (Padrão é o timestamp em ns)
```

## 📁 Estrutura do Projeto

```text
. 
├── configs/        # Arquivos YAML com configurações dos experimentos. 
├── data/           # Arquivos em sqlite ou CSV das bases de dados. 
├── src/            # Códigos-fonte principais. 
├── results/        # Resultados gerados pelos ensaios. 
├── lab.py          # Ponto de entrada do script. 
├── requirements.txt
├── README.md
└── .gitignore.
```

## ⚙️Exemplo de configuração YAML

```yaml
task: # regression | classification

data:
  source: # csv | sqlite
  path: # path da base de dados
  sqlite_table_name: # (Obrigatório em sqlite) Nome da tabela contendo os dados
  target_col: # índice da coluna da label/resultado (csv) | Nome da coluna da tabela de label/resultado (sqlite)
  features_cols:  # índices das colunas das entradas (csv) | Nomes das colunas das entradas (sqlite)
  split: # Divisão da base de dados 
    test_size: # [Proporção de treinamento, Proporção de validação, Proporção de teste,]. Soma deve ser igual a 1
    shuffle: true # Embaralhamento da base de dados durante o treinamento

model:
  input_size: # Número de features/entradas
  layers: # Configuração do número de neurônios em cada camada [camada oculta 1, camada oculta 2, ..., camada de saída]
  activation_function: # Função de ativação entre camadas ocultas. relu | sigmoide | identity | tanh
  output_activation:  # Função de ativação na camada de saída. relu | sigmoide | identity | tanh

train:
  loss: # mse | bce | bce_logits
  optimizer:
    name: # adam | sgd
    lr: # Learning rate
    betas: # (Opcional e exclusivo em adam) [beta1, beta2] 
  batch_size: # Batch Size
  epochs: # Número de épocas

experiment:
  mode: # repeat_seeds | preprocess_grid
  preprocess: # (Obrigatório em  preprocess_grid). null | minmax | standard. Pode ser um array
  seeds: # array de seeds de fixação (Obrigatório em repeat_seeds) | (Opcional em preprocess_grid) seed de fixação
```

## 🚧 Status do Projeto

Projeto em desenvolvimento, com foco experimental.  
A interface de configuração e os formatos de saída podem sofrer alterações.  