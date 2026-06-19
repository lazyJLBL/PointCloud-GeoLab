import { createRouter, createWebHistory } from 'vue-router'

import Benchmark from '../pages/Benchmark.vue'
import Dashboard from '../pages/Dashboard.vue'
import Datasets from '../pages/Datasets.vue'
import Geometry from '../pages/Geometry.vue'
import Portfolio from '../pages/Portfolio.vue'
import Preprocessing from '../pages/Preprocessing.vue'
import Primitives from '../pages/Primitives.vue'
import Registration from '../pages/Registration.vue'
import Reports from '../pages/Reports.vue'
import Segmentation from '../pages/Segmentation.vue'

export default createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/', component: Dashboard },
    { path: '/datasets', component: Datasets },
    { path: '/preprocessing', component: Preprocessing },
    { path: '/registration', component: Registration },
    { path: '/segmentation', component: Segmentation },
    { path: '/geometry', component: Geometry },
    { path: '/primitives', component: Primitives },
    { path: '/benchmark', component: Benchmark },
    { path: '/portfolio', component: Portfolio },
    { path: '/reports', component: Reports },
  ],
})
