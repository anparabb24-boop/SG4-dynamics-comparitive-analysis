function sg4_plot_example(S, P, idx, ttl, pngPath)
%SG4_PLOT_EXAMPLE Plot clean / noisy / reconstructed signals for trajectory idx.
names = {'delta','omega','eq_prime','ed_prime','id','iq','Pg','Qg'};
Cl = [S.X_clean{idx}, S.Y_clean{idx}];
Nz = [S.X_noisy{idx}, S.Y_noisy{idx}];
t  = S.t{idx}(:);

fig = figure('Visible','off','Position',[100 100 1500 1100]);
tl = tiledlayout(4,2,'TileSpacing','compact');
title(tl, ttl, 'Interpreter','none');
for k = 1:8
    nexttile;
    plot(t, Nz(:,k), 'Color', [0.75 0.75 0.75]); hold on;
    plot(t, Cl(:,k), 'k', 'LineWidth', 1.6);
    plot(t, P{idx}(:,k), 'r', 'LineWidth', 1.0);
    title(names{k}, 'Interpreter','none');
    xlabel('Time (s)'); grid on;
    if k == 1
        legend({'Noisy measurement','Clean reference','Reconstruction'}, 'Location','northeast');
    end
end
exportgraphics(fig, pngPath, 'Resolution', 150);
close(fig);
end
