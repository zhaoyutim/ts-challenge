import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import dates as mdates
from create_dataset import load_timeseries_data, drop_zero_columns



def animate(i):
    """Update the order book plot for each frame"""
    ax.clear()
    
    # Get current timestamp data
    current_data = df_filtered.iloc[i]
    timestamp = current_data['DataTime']
    
    # Prepare ask and bid data
    ask_prices = [current_data[f'AskPrice{n}'] for n in range(1, 11)]
    ask_volumes = [current_data[f'AskVolume{n}'] for n in range(1, 11)]
    bid_prices = [current_data[f'BidPrice{n}'] for n in range(1, 11)]
    bid_volumes = [current_data[f'BidVolume{n}'] for n in range(1, 11)]
    
    # Plot asks and bids
    ax.barh(ask_prices, ask_volumes, height=0.01, color='red', alpha=0.7, label='Ask')
    ax.barh(bid_prices, bid_volumes, height=0.01, color='green', alpha=0.7, label='Bid')
    
    # Formatting
    ax.set_title(f"Order Book - {timestamp}")
    ax.set_xlabel('Volume')
    ax.set_ylabel('Price')
    ax.legend()
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:,.0f}'))
    ax.grid(True)
    
    return ax



if __name__ == '__main__':
    df_filtered = load_timeseries_data(stock_symbol=["002521", "300132"])
    df_filtered = drop_zero_columns(df_filtered)
    df_filtered = df_filtered.sort_values('DataTime')  # Ensure chronological order
    # Prepare order book data
    df_filtered = df_filtered.sort_values('DataTime')  # Ensure chronological order

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(10, 6))
    # Create animation
    ani = animation.FuncAnimation(
        fig=fig,
        func=animate,
        frames=len(df_filtered),
        interval=100,  # 100ms between frames
        blit=False
    )

    # Save as GIF
    ani.save('order_book.gif', writer='pillow', dpi=100)
    plt.close()
