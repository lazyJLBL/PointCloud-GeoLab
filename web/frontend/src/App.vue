<template>
  <el-container class="app-shell">
    <el-aside width="248px" class="sidebar">
      <div class="brand">
        <strong>PointCloud-GeoLab</strong>
        <span>Experimental Web Console</span>
      </div>
      <el-menu :default-active="$route.path" router>
        <el-menu-item v-for="item in navItems" :key="item.path" :index="item.path">
          {{ item.label }}
        </el-menu-item>
      </el-menu>
    </el-aside>
    <el-container>
      <el-header class="topbar">
        <span>Python 3.10-3.12 backend</span>
        <span class="muted">Benchmark timings are local references.</span>
      </el-header>
      <el-main>
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
  border-right: 1px solid #d9e2ec;
  background: #fff;
}

.brand {
  display: grid;
  gap: 4px;
  padding: 18px 16px;
  border-bottom: 1px solid #d9e2ec;
}

.brand span {
  color: #64748b;
  font-size: 13px;
}

.topbar {
  display: flex;
  align-items: center;
  justify-content: space-between;
  border-bottom: 1px solid #d9e2ec;
  background: #fff;
}
</style>
