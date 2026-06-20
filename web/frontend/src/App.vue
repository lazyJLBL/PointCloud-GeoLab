<template>
  <el-container class="app-shell">
    <el-aside width="260px" class="sidebar">
      <div class="brand">
        <strong>PointCloud-GeoLab</strong>
        <span class="muted">Experimental Web Console</span>
      </div>
      <el-menu :default-active="$route.path" router class="nav-menu">
        <el-menu-item v-for="item in navItems" :key="item.path" :index="item.path">
          <span class="nav-label">{{ item.label }}</span>
        </el-menu-item>
      </el-menu>
    </el-aside>
    <el-container>
      <el-header class="topbar">
        <div class="topbar-content">
          <span class="status-badge">Python 3.10-3.12 backend</span>
          <span class="muted text-sm">Benchmark timings are local references only.</span>
        </div>
      </el-header>
      <el-main class="main-content">
        <router-view />
      </el-main>
    </el-container>
  </el-container>
</template>

<script setup lang="ts">
const navItems = [
  { path: '/', label: 'Dashboard' },
  { path: '/datasets', label: 'Datasets' },
  { path: '/preprocessing', label: 'Preprocessing' },
  { path: '/registration', label: 'Registration' },
  { path: '/segmentation', label: 'Segmentation' },
  { path: '/geometry', label: 'Geometry' },
  { path: '/primitives', label: 'Primitives' },
  { path: '/benchmark', label: 'Benchmark' },
  { path: '/portfolio', label: 'Portfolio' },
  { path: '/reports', label: 'Reports' },
]
</script>

<style scoped>
.app-shell {
  min-height: 100vh;
}

.sidebar {
  border-right: 1px solid #e2e8f0;
  background: #f8fafc;
  display: flex;
  flex-direction: column;
}

.brand {
  display: grid;
  gap: 4px;
  padding: 24px 20px;
  border-bottom: 1px solid #e2e8f0;
  background: #ffffff;
}

.brand strong {
  font-size: 16px;
  color: #0f172a;
}

.brand span {
  font-size: 12px;
}

.nav-menu {
  border-right: none;
  background: transparent;
  padding: 12px 0;
}

.nav-label {
  font-weight: 500;
}

.topbar {
  display: flex;
  align-items: center;
  border-bottom: 1px solid #e2e8f0;
  background: #ffffff;
  padding: 0 24px;
  height: 60px;
}

.topbar-content {
  display: flex;
  justify-content: space-between;
  align-items: center;
  width: 100%;
}

.status-badge {
  background: #f1f5f9;
  padding: 4px 10px;
  border-radius: 6px;
  font-size: 13px;
  font-weight: 500;
  color: #334155;
  border: 1px solid #e2e8f0;
}

.text-sm {
  font-size: 13px;
}

.main-content {
  background: #f8fafc;
  padding: 24px;
}
</style>
