# Proposta de Redução do Problema do Planejamento Florestal para Aplicação de Métodos Exatos

> 📄 **Documento Completo (PDF):** A monografia de conclusão de curso está disponível na íntegra no repositório: [Proposta de Redução do Problema de Planejamento Florestal para Aplicação de Métodos Exatos.pdf](./Proposta%20de%20Redução%20do%20Problema%20de%20Planejamento%20Florestal%20para%20Aplicação%20de%20Métodos%20Exatos.pdf).

## Introdução

O setor de florestas plantadas no Brasil desempenha um papel fundamental no cenário socioeconômico do país, ao contribuir com a produção de bens e serviços, agregação de valor aos produtos florestais e para a geração de empregos, divisas, tributos e renda. Em 2023, a indústria de árvores cultivadas brasileira gerou 33,4 mil novos postos de trabalho, somando um total de 2,69 milhões de empregos diretos e indiretos. Possuindo 10 milhões de hectares de árvores cultivadas, um comércio internacional de US$ 12,7 bilhões e continua sendo a maior exportadora de celulose no mundo.

Diante da importância do setor florestal para o cenário socioeconômico brasileiro, um dos maiores desafios do planejamento florestal é a elaboração de planos que possam ser adaptados à execução, através de ferramentas de otimização para assegurar a sustentabilidade no empreendimento florestal.

A coordenação no planejamento da colheita florestal permite a otimização dos fatores de produção, elevar a produtividade, reduzir custos, regular o fluxo de madeira e aprimorar o uso de equipamentos e maquinários. A programação linear vem sendo utilizada para solucionar problemas em sistemas agroindustriais e agroflorestais há bastante tempo, devido à complexidade desses sistemas e às inter-relações dos subsistemas que os compõem.

Outra ferramenta amplamente estudada e utilizada nos meios científicos são as meta-heurísticas, assim chamadas as heurísticas estocásticas, que, com o passar dos anos, também têm sido utilizadas no âmbito florestal. No entanto, uma abordagem utilizando meta-heurísticas não garante uma solução ótima global, mas permite flexibilidade na exploração do espaço de soluções do problema.

O Problema do Planejamento Florestal (PPF) apresenta uma dificuldade na obtenção de uma solução ótima através de algoritmos exatos, pois pode ser classificado como um problema da classe NP-difícil (*Nondeterministic Polynomial Time*). Problemas dessa classe possuem apenas soluções ótimas conhecidas em tempo exponencial, mas podem-se encontrar soluções não ótimas, porém boas o bastante, com algoritmos rápidos. Entretanto, pesquisas utilizando métodos exatos em problemas dessa classificação continuam sendo desenvolvidas. Em diferentes casos, essas pesquisas, mesmo que em áreas distintas, também contribuem para o PPF.

Nesse âmbito, o presente trabalho procura estabelecer uma relação entre diferentes perspectivas e aplicá-las ao PPF, visando ampliar o leque de possibilidades para a exploração do problema.

### Objetivo Geral

O objetivo principal deste trabalho é realizar a redução em tempo polinomial do PPF como uma variante do Problema da Mochila (*Knapsack Problem*), demonstrando formalmente que a análise dessa aplicação permite o uso de técnicas clássicas de otimização e algoritmos existentes na literatura para resolver essa representação do PPF.

---

## Autoria

* **Autor:** Gustavo Henrique Alves Rocha
* **Orientadora:** Profª. Drª. Luciana Balieiro Cosme
* **Coorientador:** Prof. Dr. Alberto Alexandre Assis Miranda
* **Instituição:** Instituto Federal do Norte de Minas Gerais (IFNMG) - Campus Montes Claros
* **Curso:** Bacharelado em Ciência da Computação

---

## Estrutura do Repositório

* **`benchmarks/`:** Contém as bases de instâncias clássicas da literatura (I01 a I13), instâncias controladas (5x10, 10x25, 20x50) e a instância florestal (`base120.csv`), além dos gabaritos conhecidos.
* **Códigos MMKP (Programação Dinâmica Pseudo-Polinomial):**
  * `MMKP_nao-ordenado_sem-demanda.py`: Variante Básica processando grupos na ordem original.
  * `MMKP_ordenado-peso_sem-demanda.py`: Variante Básica com ordenação prévia dos grupos por peso médio.
  * `MMKP_nao-ordenado_com-demanda.py`: Variante com filtro de demanda mínima e sem ordenação prévia.
  * `MMKP_ordenado-peso_com-demanda.py`: Variante com filtro de demanda mínima e ordenação prévia por peso.
  * `MMKP_nao-ordenado_com-penalidade-final-dp.py`: Variante com penalidade de demanda aplicada apenas no final da DP.
  * `MMKP_ordenado-peso_com-penalidade-final-dp.py`: Variante com penalidade aplicada no final e ordenação prévia por peso.
  * `MMKP_nao-ordenado_com-penalidade-aplicada.py`: Variante com penalização *in-process* durante a construção dos estados na DP.
  * `MMKP_ordenado-peso_com-penalidade-aplicada.py`: Variante com penalização *in-process* e ordenação prévia por peso.
  * `MMKP_nao-ordenado_chaveamento.py`: Variante com alternância dinâmica entre lucro bruto e penalização durante a DP.
  * `MMKP_ordenado-peso_chaveamento.py`: Variante com chaveamento e ordenação prévia por peso.
  * `MMKP_nao-ordenado_hibrido.py`: Variante híbrida mantendo rastreamento de lucro bruto e penalizado.
  * `MMKP_ordenado-peso_hibrido.py`: Variante híbrida com ordenação prévia por peso.
* **Meta-heurística Simulated Annealing:**
  * `MMKP_Recozimento_Simulado.py`: Implementação do algoritmo de Recozimento Simulado adaptado ao PPF/MMKP.
* **Otimizadores com Optuna:**
  * `optuna_tunneling_mmkp.py`: Otimização Bayesiana de hiperparâmetros (escala, limite, heurística) para o algoritmo de Programação Dinâmica.
  * `optuna_tunneling_sa.py`: Otimização Bayesiana de hiperparâmetros (temperatura inicial, taxa de resfriamento, iterações) para o Recozimento Simulado.
* **Pastas de Resultados:** Contêm rigorosamente os arquivos `.json` das soluções experimentais citadas nas tabelas e anexos da dissertação.
* **Documento Completo:** [Proposta de Redução do Problema de Planejamento Florestal para Aplicação de Métodos Exatos.pdf](./Proposta%20de%20Redução%20do%20Problema%20de%20Planejamento%20Florestal%20para%20Aplicação%20de%20Métodos%20Exatos.pdf) (Arquivo `.pdf` com o texto integral do trabalho).

---

## Requisitos e Instalação

Recomenda-se utilizar Python 3.10 ou superior. As dependências necessárias podem ser instaladas via `pip`:

```bash
pip install numpy pandas optuna
```

---

## Como Executar os Códigos

### 1. Executando os Algoritmos MMKP (Programação Dinâmica)

Cada algoritmo MMKP lê um arquivo de configuração de testes com extensão `.in`. O arquivo de teste especifica quais bases, heurísticas e parâmetros serão executados em lote.

Para executar, basta rodar:

```bash
python3 MMKP_ordenado-peso_com-demanda.py teste.in
```

#### Formato do Arquivo de Teste (`.in`)

O arquivo deve iniciar com a palavra-chave `ini` em uma linha isolada. Cada linha subsequente define um teste:

**Para as Variantes Básica, Com Demanda, Penalidade Final e Penalidade Aplicada:**
```text
<base> <heuristica> <scale_vpl> <limit> <buffer> [extra]
```
* `<base>`: Identificador da base presente em `benchmarks/` (`5x10`, `10x25`, `20x50`, `I01` a `I13`, `base120`).
* `<heuristica>`: Sigla da heurística de truncamento (`N`, `A`, `MG`, `MGA`, `MGR`, `MM`, `DV`).
* `<scale_vpl>`: Fator de escalonamento do lucro (ex: `25000`, `50000`, `100000`).
* `<limit>`: Limite de estados preservados por nível de lucro (ex: `50`, `100`).
* `<buffer>`: Folga de capacidade (usualmente `1.0`).
* `[extra]`: Parâmetro adicional referente à heurística:
  * Para a heurística `A` (Pareto Avançado): `0.1` (fração de soluções mantidas por dimensão).
  * Para as heurísticas `MG`, `MGA`, `MGR`, `MM`: `C` (ordem crescente de ocupação) ou `D` (ordem decrescente).
  * Para as heurísticas `N` e `DV`: parâmetro omitido.

*Exemplo de `teste.in`:*
```text
ini
base120 N 25000 50 1.0
base120 A 50000 50 1.0 0.1
base120 MG 50000 50 1.0 C
base120 DV 25000 100 1.0
```

**Para as Variantes de Chaveamento e Híbrida:**
Possuem o parâmetro adicional `<cut_val>` (ponto de corte da transição de fase):
```text
<base> <heuristica> <scale_vpl> <limit> <buffer> <cut_val> [extra]
```
* `<cut_val>`: `1` (100% bruto), `2` (transição em 50% dos grupos), `3` (transição em 33%), etc.

*Exemplo de `teste_chaveamento.in`:*
```text
ini
base120 N 25000 50 1.0 2
base120 A 25000 50 1.0 2 0.1
base120 MG 50000 50 1.0 2 C
```

---

### 2. Executando o Recozimento Simulado (Simulated Annealing)

O código `MMKP_Recozimento_Simulado.py` pode ser executado diretamente com parâmetros padrão ou passando argumentos específicos via linha de comando:

```bash
# Execução padrão na Base 120:
python3 MMKP_Recozimento_Simulado.py

# Execução com parâmetros personalizados:
# Formato: python3 MMKP_Recozimento_Simulado.py <alvo> <temp_ini> <taxa_resf> <max_iter_temp> <total_iter>
python3 MMKP_Recozimento_Simulado.py base120 100000 0.990 40 50000
```
Os resultados gerados são salvos automaticamente na pasta `resultados_recozimento-simulado/`.

---

### 3. Executando os Otimizadores Autônomos com Optuna

Os scripts Optuna realizam tunelamento bayesiano de hiperparâmetros com suporte a timeout e restrições de viabilidade:

* **Para otimização da Programação Dinâmica (MMKP):**
  ```bash
  python3 optuna_tunneling_mmkp.py
  ```
  O script otimiza a escolha da variante, heurística, `scale_vpl`, `limit` e `extra_alpha`, salvando o histórico no banco SQLite `optuna_mmkp.db`.

* **Para otimização do Recozimento Simulado (SA):**
  ```bash
  python3 optuna_tunneling_sa.py
  ```
  O script busca automaticamente a melhor combinação de temperatura inicial, taxa de resfriamento e quantidade de iterações, salvando o progresso em `optuna_mmkp_sa.db`.
