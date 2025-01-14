# Projeto de Treinamento de Drone com Cosmos

## Visão Geral
Este projeto implementa um sistema de treinamento de drone usando o Cosmos SDK e um modelo de LLM para controle drone através de texto, chamado SOVA. O sistema utiliza o modelo Cosmos-1.0-Diffusion-7B-Text2World da NVIDIA para interpretar comandos textuais e gerar ações apropriadas do drone em um ambiente simulado a ser replicado posteriormente para um sistema real de segurança remota e assitida.

## Funcionalidades
- Sistema de controle de drone baseado em texto
- Integração com o modelo Cosmos da NVIDIA
- Simulação de ambiente em tempo real
- Pipeline de treinamento automatizado
- Salvamento de checkpoints do modelo
- Análise personalizada de comandos para o drone

## Requisitos
- Python 3.x
- PyTorch
- Cosmos SDK
- Biblioteca Transformers
- GPU compatível com CUDA (recomendado)

## Instalação
```bash
pip install cosmos_sdk torch transformers
```

## Estrutura do Projeto
O projeto consiste em dois componentes principais:
1. **Classe DroneAgent**: Gerencia a interação entre o modelo de linguagem e as ações do drone
2. **Loop de Treinamento**: Administra o processo de treinamento ao longo de múltiplas épocas

### DroneAgent
A classe DroneAgent é responsável por:
- Processar observações do ambiente
- Gerar ações baseadas em entrada de texto
- Analisar comandos textuais em sinais de controle do drone
- Salvar checkpoints do modelo

### Pipeline de Treinamento
O sistema de treinamento:
- Executa por um número específico de épocas
- Coleta feedback do ambiente
- Calcula recompensas
- Atualiza o modelo
- Salva o progresso periodicamente

## Interpretação de Comandos
O sistema atualmente suporta os seguintes comandos:
- "subir"
- "descer"
- "virar à esquerda"
- "virar à direita"

## Como Usar
Para executar o processo de treinamento:

```python
# Inicializar ambiente e agente
env = CosmosEnvironment(config='base/cosmos_config.yaml')
drone = DroneAgent(env, model=model, tokenizer=tokenizer)

# Iniciar treinamento
train_drone(drone, env, epochs=20)
```

## Detalhes do Modelo
- Modelo Base: nvidia/Cosmos-1.0-Diffusion-7B-Text2World
- Arquitetura: Baseada em Transformer
- Entrada: Observações em texto
- Saída: Comandos de controle do drone

## Processo de Treinamento
1. Reinicialização do ambiente no início de cada época
2. Loop contínuo de ação-observação
3. Acúmulo de recompensas
4. Salvamento de checkpoint do modelo após cada época
5. Renderização do ambiente para visualização

## Salvamento do Modelo
Os checkpoints são salvos após cada época com o padrão de nomenclatura:
```
drone_model_epoch_[numero_da_epoca].pth
```
