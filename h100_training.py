# -*- coding: utf-8 -*-
"""
고급 학습 모니터링 도구들 (선택사항)
메인 코드에 추가하거나 별도로 사용 가능
"""

import matplotlib.pyplot as plt
import seaborn as sns
from IPython.display import clear_output
import wandb  # pip install wandb
import time
import threading
import os

# -------------------------------
# 1) 실시간 그래프 플롯 (Jupyter/Colab용)
# -------------------------------
class RealTimePlotter:
    def __init__(self):
        plt.ion()  # 대화형 모드 켜기
        self.fig, self.axes = plt.subplots(2, 2, figsize=(15, 10))
        self.fig.suptitle('Real-time Training Monitor', fontsize=16)
        
    def update_plots(self, train_losses, train_accs, val_losses, val_accs, 
                    epoch_times, best_val_acc, current_epoch):
        clear_output(wait=True)
        
        epochs = list(range(len(train_losses)))
        
        # Loss 그래프
        self.axes[0,0].clear()
        self.axes[0,0].plot(epochs, train_losses, 'b-', label='Train Loss', alpha=0.7)
        self.axes[0,0].plot(epochs, val_losses, 'r-', label='Val Loss', alpha=0.7)
        self.axes[0,0].set_title('Loss Over Time')
        self.axes[0,0].set_xlabel('Epoch')
        self.axes[0,0].set_ylabel('Loss')
        self.axes[0,0].legend()
        self.axes[0,0].grid(True, alpha=0.3)
        
        # Accuracy 그래프
        self.axes[0,1].clear()
        self.axes[0,1].plot(epochs, train_accs, 'b-', label='Train Acc', alpha=0.7)
        self.axes[0,1].plot(epochs, val_accs, 'r-', label='Val Acc', alpha=0.7)
        self.axes[0,1].axhline(y=best_val_acc, color='g', linestyle='--', 
                              label=f'Best Val: {best_val_acc:.2f}%')
        self.axes[0,1].set_title('Accuracy Over Time')
        self.axes[0,1].set_xlabel('Epoch')
        self.axes[0,1].set_ylabel('Accuracy (%)')
        self.axes[0,1].legend()
        self.axes[0,1].grid(True, alpha=0.3)
        
        # 에포크별 시간
        self.axes[1,0].clear()
        self.axes[1,0].plot(epochs, epoch_times, 'g-', alpha=0.7)
        if len(epoch_times) > 10:
            moving_avg = np.convolve(epoch_times, np.ones(10)/10, mode='valid')
            self.axes[1,0].plot(epochs[9:], moving_avg, 'orange', linewidth=2, 
                              label='Moving Average (10)')
            self.axes[1,0].legend()
        self.axes[1,0].set_title('Training Speed')
        self.axes[1,0].set_xlabel('Epoch')
        self.axes[1,0].set_ylabel('Time per Epoch (s)')
        self.axes[1,0].grid(True, alpha=0.3)
        
        # 학습률
        self.axes[1,1].clear()
        if hasattr(self, 'learning_rates'):
            self.axes[1,1].semilogy(epochs, self.learning_rates, 'purple', alpha=0.7)
            self.axes[1,1].set_title('Learning Rate Schedule')
            self.axes[1,1].set_xlabel('Epoch')
            self.axes[1,1].set_ylabel('Learning Rate')
            self.axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()

# -------------------------------
# 2) Weights & Biases 연동 (고급 실험 추적)
# -------------------------------
def setup_wandb(project_name="indoor_positioning", experiment_name=None):
    """
    W&B 설정 - 클라우드에서 실험 추적
    사용법: wandb login 후 이 함수 호출
    """
    config = {
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "hidden_sizes": HIDDEN_SIZES,
        "dropout": DROPOUT,
        "weight_decay": WEIGHT_DECAY,
        "max_epochs": MAX_EPOCHS,
        "architecture": "MLP",
        "dataset_size": len(train_loader.dataset)
    }
    
    wandb.init(
        project=project_name,
        name=experiment_name,
        config=config
    )
    return wandb

def log_to_wandb(epoch, train_loss, train_acc, val_loss, val_acc, 
                learning_rate, epoch_time):
    """W&B에 메트릭 로깅"""
    wandb.log({
        "epoch": epoch,
        "train/loss": train_loss,
        "train/accuracy": train_acc,
        "val/loss": val_loss,
        "val/accuracy": val_acc,
        "learning_rate": learning_rate,
        "epoch_time": epoch_time
    })

# -------------------------------
# 3) 텔레그램 알림 (선택사항)
# -------------------------------
def send_telegram_notification(message, bot_token=None, chat_id=None):
    """
    텔레그램으로 학습 진행 상황 알림
    봇 토큰과 채팅 ID 필요
    """
    if not bot_token or not chat_id:
        return
        
    try:
        import requests
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        data = {
            "chat_id": chat_id,
            "text": message,
            "parse_mode": "HTML"
        }
        requests.post(url, data=data, timeout=10)
    except:
        pass  # 실패해도 학습은 계속

# -------------------------------
# 4) 향상된 학습 루프 (위 기능들 통합)
# -------------------------------
def enhanced_training_loop(model, train_loader, val_loader, criterion, optimizer, 
                          scheduler, device, max_epochs=200, patience=25,
                          use_wandb=False, use_telegram=False, use_plots=False,
                          telegram_config=None):
    """
    모든 모니터링 기능이 통합된 학습 루프
    """
    
    # 모니터링 도구 초기화
    plotter = RealTimePlotter() if use_plots else None
    
    if use_wandb:
        wandb_run = setup_wandb()
    
    # 학습 변수들
    best_val_acc = 0.0
    patience_counter = 0
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    learning_rates = []
    epoch_times = []
    improvement_epochs = []
    
    start_time = time.time()
    
    # 시작 알림
    if use_telegram and telegram_config:
        send_telegram_notification(
            f"🚀 <b>Training Started</b>\n"
            f"Model: MLP ({sum(p.numel() for p in model.parameters()):,} params)\n"
            f"Dataset: {len(train_loader.dataset):,} samples\n"
            f"Max Epochs: {max_epochs}",
            **telegram_config
        )
    
    print(f"{'Epoch':>5} | {'Train Loss':>10} {'Train Acc':>10} | {'Val Loss':>8} {'Val Acc':>8} | {'LR':>8} | {'Time':>6} | {'Best':>6}")
    print("-" * 85)
    
    for epoch in range(max_epochs):
        epoch_start = time.time()
        
        # 학습 & 검증
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # 스케줄러 업데이트
        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']
        
        # 시간 기록
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        
        # 메트릭 저장
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        learning_rates.append(current_lr)
        
        # 베스트 모델 체크
        improved = val_acc > best_val_acc
        if improved:
            best_val_acc = val_acc
            patience_counter = 0
            improvement_epochs.append(epoch)
            
            # 모델 저장
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_val_acc': best_val_acc,
                # ... 다른 정보들
            }, "best_model.pth")
            
        else:
            patience_counter += 1
        
        # W&B 로깅
        if use_wandb:
            log_to_wandb(epoch, train_loss, train_acc, val_loss, val_acc, 
                        current_lr, epoch_time)
        
        # 실시간 플롯 업데이트
        if use_plots and plotter and (epoch % 5 == 0 or improved):
            plotter.learning_rates = learning_rates
            plotter.update_plots(train_losses, train_accs, val_losses, val_accs,
                               epoch_times, best_val_acc, epoch)
        
        # 콘솔 출력
        show_epoch = (epoch < 20 or epoch % 5 == 0 or improved or epoch == max_epochs-1)
        if show_epoch:
            status = "🎯 NEW!" if improved else ""
            print(f"{epoch:5d} | {train_loss:10.4f} {train_acc:9.2f}% | "
                  f"{val_loss:8.4f} {val_acc:7.2f}% | {current_lr:8.2e} | "
                  f"{epoch_time:5.1f}s | {best_val_acc:5.2f}% {status}")
        
        # 중요한 이정표에서 텔레그램 알림
        if use_telegram and telegram_config:
            should_notify = (
                improved and val_acc > 85.0 or  # 높은 성능 달성
                epoch % 100 == 0 or  # 100에포크마다
                patience_counter == patience // 2  # 절반 patience 도달
            )
            
            if should_notify:
                elapsed = time.time() - start_time
                send_telegram_notification(
                    f"📊 <b>Training Update</b> (Epoch {epoch})\n"
                    f"Val Accuracy: {val_acc:.2f}% {'🎯' if improved else ''}\n"
                    f"Best So Far: {best_val_acc:.2f}%\n"
                    f"Elapsed: {elapsed/60:.1f}min\n"
                    f"Patience: {patience_counter}/{patience}",
                    **telegram_config
                )
        
        # 조기 종료
        if patience_counter >= patience:
            break
    
    # 최종 알림
    training_time = time.time() - start_time
    if use_telegram and telegram_config:
        send_telegram_notification(
            f"✅ <b>Training Completed!</b>\n"
            f"Best Accuracy: {best_val_acc:.2f}%\n"
            f"Total Time: {training_time/60:.1f}min\n"
            f"Epochs: {len(train_losses)}",
            **telegram_config
        )
    
    return {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'val_losses': val_losses,
        'val_accs': val_accs,
        'best_val_acc': best_val_acc,
        'training_time': training_time
    }

# -------------------------------
# 사용 예시
# -------------------------------
"""
# 기본 사용법 (메인 코드에서)
results = enhanced_training_loop(
    model, train_loader, val_loader, criterion, optimizer, scheduler, device,
    max_epochs=200, patience=25
)

# W&B 사용 시
results = enhanced_training_loop(
    model, train_loader, val_loader, criterion, optimizer, scheduler, device,
    use_wandb=True
)

# 텔레그램 알림 사용 시
telegram_config = {
    'bot_token': 'YOUR_BOT_TOKEN',
    'chat_id': 'YOUR_CHAT_ID'
}
results = enhanced_training_loop(
    model, train_loader, val_loader, criterion, optimizer, scheduler, device,
    use_telegram=True, telegram_config=telegram_config
)

# 실시간 플롯 사용 시 (Jupyter/Colab)
results = enhanced_training_loop(
    model, train_loader, val_loader, criterion, optimizer, scheduler, device,
    use_plots=True
)
"""