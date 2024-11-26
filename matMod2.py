import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Параметры модели
N = 100 # Размер сетки
timesteps = 200  # Количество временных шагов
p_initial = 0.2  # Начальная вероятность заселения клетки

# Параметры взаимодействия
N_overcrowd, N_underpop, N_reproduce = 4, 2, 3
mutation_rate = 0.01  # Вероятность мутации клетки
external_disturbance_rate = 0.05  # Вероятность внешнего воздействия

# Начальная сетка
grid = np.random.choice([0, 1], size=(N, N), p=[1 - p_initial, p_initial])
population_counts = []  # Для записи численности популяции


# Функция для подсчёта соседей
def count_neighbors(grid):
    neighbors = (
            np.roll(np.roll(grid, 1, axis=0), 1, axis=1) +
            np.roll(grid, 1, axis=0) +
            np.roll(np.roll(grid, 1, axis=0), -1, axis=1) +
            np.roll(grid, 1, axis=1) +
            np.roll(grid, -1, axis=1) +
            np.roll(np.roll(grid, -1, axis=0), 1, axis=1) +
            np.roll(grid, -1, axis=0) +
            np.roll(np.roll(grid, -1, axis=0), -1, axis=1)
    )
    return neighbors


# Обновление состояния сетки
def update(grid, frame):
    new_grid = grid.copy()
    neighbor_counts = count_neighbors(grid)  # Плотность соседей
    for i in range(N):
        for j in range(N):
            neighbors = neighbor_counts[i, j]
            if grid[i, j] == 1:  # Если клетка жива
                if neighbors < N_underpop or neighbors > N_overcrowd:
                    new_grid[i, j] = 0
            else:  # Если клетка пуста
                if neighbors == N_reproduce:
                    new_grid[i, j] = 1

            # Применение мутации
            if np.random.random() < mutation_rate:
                new_grid[i, j] = 1 - grid[i, j]

    # Внешнее воздействие каждые несколько шагов
    if frame % 10 == 0:  # Возмущение каждые 10 шагов
        disturbances = int(external_disturbance_rate * N * N)  # Количество возмущений
        for _ in range(disturbances):
            x, y = np.random.randint(0, N, size=2)
            new_grid[x, y] = 1  # Заселение случайных клеток

    return new_grid, neighbor_counts


# Визуализация
fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
img = ax1.imshow(grid, cmap="viridis", interpolation="nearest")
ax1.set_title("Модель клеточного автомата\n(Состояние клеток)")
ax1.set_xlabel("Координата X")
ax1.set_ylabel("Координата Y")

density_map = ax2.imshow(np.zeros_like(grid), cmap="coolwarm", interpolation="nearest", vmin=0, vmax=8)
ax2.set_title("Карта плотности соседей\n(Число соседей для каждой клетки)")
ax2.set_xlabel("Координата X")
ax2.set_ylabel("Координата Y")
colorbar = plt.colorbar(density_map, ax=ax2, shrink=0.8)
colorbar.set_label("Число соседей")

ax3.set_xlim(0, timesteps)
ax3.set_ylim(0, N * N)
line, = ax3.plot([], [], lw=2, label="Число живых клеток")
ax3.legend()
ax3.set_title("Динамика популяции\n(Число живых клеток со временем)")
ax3.set_xlabel("Шаги времени")
ax3.set_ylabel("Число живых клеток")

# Аннотации на каждой панели
annotations = []


def add_annotations(frame, live_cells):
    global annotations
    # Удаляем предыдущие аннотации
    for annotation in annotations:
        annotation.remove()
    annotations.clear()

    # Добавляем новые аннотации
    annotations.append(ax1.annotate(f"Шаг: {frame}", (1, 1), color="white", fontsize=10, ha="left", va="top",
                                    bbox=dict(facecolor="black", alpha=0.5)))
    annotations.append(ax2.annotate("Красный: высокая плотность\nСиний: низкая плотность", (0, 0),
                                    color="white", fontsize=10, ha="left", va="top",
                                    bbox=dict(facecolor="black", alpha=0.5)))
    annotations.append(ax3.annotate(f"Живые клетки: {live_cells}", (timesteps * 0.7, N * N * 0.9),
                                    color="blue", fontsize=10, ha="center", bbox=dict(facecolor="white", alpha=0.8)))


def animate(frame):
    global grid, population_counts
    grid, neighbor_counts = update(grid, frame)
    live_cells = np.sum(grid)
    population_counts.append(live_cells)  # Считаем живые клетки
    img.set_data(grid)
    density_map.set_data(neighbor_counts)  # Обновляем карту плотности
    line.set_data(range(len(population_counts)), population_counts)

    # Добавляем аннотации
    add_annotations(frame, live_cells)

    return img, density_map, line


ani = animation.FuncAnimation(fig, animate, frames=timesteps, interval=300, blit=False)
plt.show()
