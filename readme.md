# Projeto de Treinamento de Drone com Cosmos

## Visão Geral
Este projeto implementa um sistema chamado SOVA, com a finalidade de treinamento de drone usando o NVIDIA Cosmos SDK e um modelo de LLM para controle drone através de texto. O sistema utiliza o modelo Cosmos-1.0-Diffusion-7B-Text2World da NVIDIA para interpretar comandos textuais e gerar ações apropriadas do drone em um ambiente simulado. A nova funcionalidade Perestroika permite uma reconfiguração dinâmica do sistema durante a execução.

## Funcionalidades
- Sistema de controle de drone baseado em texto
- Integração com o modelo Cosmos da NVIDIA
- Simulação de ambiente em tempo real
- Pipeline de treinamento automatizado
- Salvamento de checkpoints do modelo
- Análise personalizada de comandos para o drone
- Sistema Perestroika para reconfiguração dinâmica dos parâmetros de segurança e controle

## Requisitos
- Python 3.x
- PyTorch
- Cosmos SDK
- Biblioteca Transformers
- GPU compatível com CUDA (recomendado)
- Jetson GPIO (para controle de hardware)

## Instalação
```bash
pip install cosmos_sdk torch transformers Jetson.GPIO
```

## Estrutura do Projeto
O projeto consiste em três componentes principais:
1. **Classe DroneAgent**: Gerencia a interação entre o modelo de linguagem e as ações do drone
2. **Loop de Treinamento**: Administra o processo de treinamento ao longo de múltiplas épocas
3. **Sistema Perestroika**: Permite a reconfiguração dinâmica dos parâmetros de segurança e controle

### DroneAgent
A classe DroneAgent é responsável por:
- Processar observações do ambiente
- Gerar ações baseadas em entrada de texto
- Analisar comandos textuais em sinais de controle do drone
- Gerenciar sistemas de segurança e detecção
- Salvar checkpoints do modelo

### Pipeline de Treinamento
O sistema de treinamento:
- Executa por um número específico de épocas
- Coleta feedback do ambiente
- Calcula recompensas
- Atualiza o modelo
- Salva o progresso periodicamente
- Integra com o sistema Perestroika para ajustes dinâmicos

### Sistema Perestroika
O módulo Perestroika oferece:
- Reconfiguração dinâmica dos parâmetros de segurança
- Ajuste em tempo real dos limiares de detecção
- Modificação dos protocolos de resposta
- Personalização das ações de controle

## Interpretação de Comandos
O sistema suporta os seguintes comandos:
- "subir"
- "descer"
- "virar à esquerda"
- "virar à direita"
- Comandos personalizados via Perestroika

## Como Usar
Para executar o processo de treinamento:
```python
# Inicializar ambiente e agente
env = CosmosEnvironment(config='base/cosmos_config.yaml')
drone = DroneAgent(env, model=model, tokenizer=tokenizer)

# Configurar Perestroika (opcional)
drone.configure_perestroika(sensitivity=0.8, response_threshold=0.6)

# Iniciar treinamento
train_drone(drone, env, epochs=20)
```

## Detalhes do Modelo
- Modelo Base: nvidia/Cosmos-1.0-Diffusion-7B-Text2World
- Arquitetura: Baseada em Transformer
- Entrada: Observações em texto
- Saída: Comandos de controle do drone
- Integração com Perestroika para ajustes dinâmicos

## Processo de Treinamento
1. Reinicialização do ambiente no início de cada época
2. Loop contínuo de ação-observação
3. Acúmulo de recompensas
4. Ajustes dinâmicos via Perestroika
5. Salvamento de checkpoint do modelo após cada época
6. Renderização do ambiente para visualização

## Salvamento do Modelo
Os checkpoints são salvos após cada época com o padrão de nomenclatura:
```
drone_model_epoch_[numero_da_epoca].pth
```

## Configuração do Sistema Perestroika
O sistema Perestroika pode ser configurado através de um arquivo YAML ou dinamicamente durante a execução:
```yaml
perestroika:
  sensitivity: 0.8
  response_threshold: 0.6
  detection_mode: "advanced"
  safety_protocols: "standard"
```

## Considerações de Segurança
- O sistema inclui protocolos de segurança configuráveis
- A funcionalidade Perestroika permite ajustes dos parâmetros de segurança em tempo real
- Recomenda-se revisão periódica das configurações de segurança
- Sistema de logs para auditoria de todas as ações executadas
