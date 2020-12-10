from .base import _Iter_LRScheduler
from .registry import LR_SCHEDULERS


@LR_SCHEDULERS.register_module
class RestartPolyLR(_Iter_LRScheduler):
    """Cosine annealing with restarts learning rate scheme.
    Args:
        periods (list[int]): Periods for each cosine anneling cycle.
        restart_weights (list[float], optional): Restart weights at each
            restart iteration. Default: [1].
        min_lr (float, optional): The minimum lr. Default: None.
        min_lr_ratio (float, optional): The ratio of minimum lr to the base lr.
            Either `min_lr` or `min_lr_ratio` should be specified.
            Default: None.
    """

    def __init__(self,
                 optimizer,
                 periods,
                 niter_per_epoch,
                 restart_weights=[1],
                 power=0.9,
                 last_iter=-1):
        self.periods = periods
        self.niter_per_epoch = niter_per_epoch
        self.restart_weights = restart_weights
        self.power = power

        self.cumulative_periods = [
            sum(self.periods[0:i + 1]) for i in range(0, len(self.periods))
        ]

        assert (len(self.periods) == len(self.restart_weights)
                ), 'periods and restart_weights should have the same length.'
        super(RestartPolyLR, self).__init__(optimizer, niter_per_epoch, last_iter)

    def get_lr(self):
        idx = get_position_from_periods(self.last_epoch, self.cumulative_periods)
        current_weight = self.restart_weights[idx]
        nearest_restart = 0 if idx == 0 else self.cumulative_periods[idx - 1]
        current_periods = self.periods[idx]
        current_periods_max_iters = current_periods * self.niter_per_epoch

        last_iter = self.last_iter if idx == 0 else self.last_iter - self.cumulative_periods[idx - 1] * self.niter_per_epoch

        multiplier = (1 - last_iter / float(
            current_periods_max_iters)) ** self.power
        return [base_lr * multiplier for base_lr in self.base_lrs]


def get_position_from_periods(iteration, cumulative_periods):
    """Get the position from a period list.
    It will return the index of the right-closest number in the period list.
    For example, the cumulative_periods = [100, 200, 300, 400],
    if iteration == 50, return 0;
    if iteration == 210, return 2;
    if iteration == 300, return 3.
    Args:
        iteration (int): Current iteration.
        cumulative_periods (list[int]): Cumulative period list.
    Returns:
        int: The position of the right-closest number in the period list.
    """
    for i, period in enumerate(cumulative_periods):
        if iteration < period:
            return i
    raise ValueError(f'Current iteration {iteration} exceeds '
                     f'cumulative_periods {cumulative_periods}')
