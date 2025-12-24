use macroquad::prelude::*;
use macroquad::rand as mqrand;

// ===== CONSTANTS =====
const TILE_SIZE: f32 = 32.0;
const ROOM_WIDTH: usize = 13;
const ROOM_HEIGHT: usize = 9;
const SCREEN_WIDTH: f32 = TILE_SIZE * ROOM_WIDTH as f32;
const SCREEN_HEIGHT: f32 = TILE_SIZE * ROOM_HEIGHT as f32 + 64.0; // Extra for HUD

// ===== GAME STATE =====
#[derive(Clone, Copy, PartialEq)]
enum GameMode {
    Human,
    RL,
}

#[derive(Clone, Copy, PartialEq)]
enum TileType {
    Floor,
    Wall,
    Door(Direction),
    Rock,
    Pit,
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Direction {
    Up,
    Down,
    Left,
    Right,
}

#[derive(Clone)]
struct Player {
    x: f32,
    y: f32,
    vx: f32,
    vy: f32,
    hp: i32,
    max_hp: i32,
    damage: f32,
    speed: f32,
    fire_rate: f32,
    fire_timer: f32,
    invincible_timer: f32,
    tears_speed: f32,
}

impl Player {
    fn new() -> Self {
        Self {
            x: SCREEN_WIDTH / 2.0,
            y: (SCREEN_HEIGHT - 64.0) / 2.0 + 32.0,
            vx: 0.0,
            vy: 0.0,
            hp: 6,  // 3 hearts (half hearts)
            max_hp: 6,
            damage: 1.0,
            speed: 150.0,
            fire_rate: 0.3,
            fire_timer: 0.0,
            invincible_timer: 0.0,
            tears_speed: 300.0,
        }
    }

    fn rect(&self) -> Rect {
        Rect::new(self.x - 12.0, self.y - 12.0, 24.0, 24.0)
    }
}

#[derive(Clone, Copy, PartialEq)]
enum EnemyType {
    Fly,        // Random movement, contact damage
    Gaper,      // Chases player slowly
    Shooter,    // Stationary, shoots
    Boss,       // Big, shoots patterns
}

#[derive(Clone)]
struct Enemy {
    x: f32,
    y: f32,
    vx: f32,
    vy: f32,
    hp: f32,
    max_hp: f32,
    damage: f32,
    enemy_type: EnemyType,
    timer: f32,
    shoot_timer: f32,
}

impl Enemy {
    fn new(x: f32, y: f32, enemy_type: EnemyType) -> Self {
        let (hp, damage) = match enemy_type {
            EnemyType::Fly => (2.0, 1.0),
            EnemyType::Gaper => (4.0, 1.0),
            EnemyType::Shooter => (3.0, 0.0),
            EnemyType::Boss => (50.0, 2.0),
        };
        Self {
            x,
            y,
            vx: 0.0,
            vy: 0.0,
            hp,
            max_hp: hp,
            damage,
            enemy_type,
            timer: mqrand::gen_range(0.0, 2.0),
            shoot_timer: 0.0,
        }
    }

    fn rect(&self) -> Rect {
        let size = match self.enemy_type {
            EnemyType::Boss => 48.0,
            _ => 20.0,
        };
        Rect::new(self.x - size / 2.0, self.y - size / 2.0, size, size)
    }
}

#[derive(Clone)]
struct Tear {
    x: f32,
    y: f32,
    vx: f32,
    vy: f32,
    damage: f32,
    from_player: bool,
    lifetime: f32,
}

impl Tear {
    fn rect(&self) -> Rect {
        Rect::new(self.x - 6.0, self.y - 6.0, 12.0, 12.0)
    }
}

#[derive(Clone, Copy, PartialEq)]
enum ItemType {
    Heart,
    DamageUp,
    SpeedUp,
    TearUp,
}

#[derive(Clone)]
struct Item {
    x: f32,
    y: f32,
    item_type: ItemType,
}

#[derive(Clone)]
struct Room {
    tiles: [[TileType; ROOM_WIDTH]; ROOM_HEIGHT],
    enemies: Vec<Enemy>,
    items: Vec<Item>,
    cleared: bool,
    x: i32,  // Position in dungeon grid
    y: i32,
}

impl Room {
    fn new(x: i32, y: i32) -> Self {
        let mut tiles = [[TileType::Floor; ROOM_WIDTH]; ROOM_HEIGHT];

        // Walls
        for i in 0..ROOM_WIDTH {
            tiles[0][i] = TileType::Wall;
            tiles[ROOM_HEIGHT - 1][i] = TileType::Wall;
        }
        for j in 0..ROOM_HEIGHT {
            tiles[j][0] = TileType::Wall;
            tiles[j][ROOM_WIDTH - 1] = TileType::Wall;
        }

        Self {
            tiles,
            enemies: Vec::new(),
            items: Vec::new(),
            cleared: false,
            x,
            y,
        }
    }

    fn add_door(&mut self, dir: Direction) {
        let (tx, ty) = match dir {
            Direction::Up => (ROOM_WIDTH / 2, 0),
            Direction::Down => (ROOM_WIDTH / 2, ROOM_HEIGHT - 1),
            Direction::Left => (0, ROOM_HEIGHT / 2),
            Direction::Right => (ROOM_WIDTH - 1, ROOM_HEIGHT / 2),
        };
        self.tiles[ty][tx] = TileType::Door(dir);
    }

    fn add_obstacles(&mut self) {
        let num_rocks = mqrand::gen_range(0, 4);
        for _ in 0..num_rocks {
            let x = mqrand::gen_range(2, ROOM_WIDTH - 2);
            let y = mqrand::gen_range(2, ROOM_HEIGHT - 2);
            if self.tiles[y][x] == TileType::Floor {
                self.tiles[y][x] = if mqrand::gen_range(0.0, 1.0) < 0.3 {
                    TileType::Pit
                } else {
                    TileType::Rock
                };
            }
        }
    }

    fn spawn_enemies(&mut self, difficulty: i32) {
        let num_enemies = mqrand::gen_range(2, 4 + difficulty as usize);
        for _ in 0..num_enemies {
            let x = mqrand::gen_range(2, ROOM_WIDTH - 2) as f32 * TILE_SIZE + TILE_SIZE / 2.0;
            let y = mqrand::gen_range(2, ROOM_HEIGHT - 2) as f32 * TILE_SIZE + TILE_SIZE / 2.0 + 32.0;

            let roll = mqrand::gen_range(0, 100);
            let enemy_type = if roll <= 40 {
                EnemyType::Fly
            } else if roll <= 70 {
                EnemyType::Gaper
            } else {
                EnemyType::Shooter
            };
            self.enemies.push(Enemy::new(x, y, enemy_type));
        }
    }
}

struct Dungeon {
    rooms: Vec<Room>,
    current_room: usize,
    floor: i32,
}

impl Dungeon {
    fn generate(floor: i32) -> Self {
        let num_rooms = 5 + floor as usize * 2;
        let mut rooms = Vec::new();
        let mut positions: Vec<(i32, i32)> = vec![(0, 0)];

        // Starting room
        let mut start_room = Room::new(0, 0);
        start_room.cleared = true;
        rooms.push(start_room);

        // Generate connected rooms
        while rooms.len() < num_rooms {
            let parent_idx = mqrand::gen_range(0, positions.len());
            let (px, py) = positions[parent_idx];

            let dirs = [(0, -1), (0, 1), (-1, 0), (1, 0)];
            let (dx, dy) = dirs[mqrand::gen_range(0, 4)];
            let (nx, ny) = (px + dx, py + dy);

            if positions.contains(&(nx, ny)) {
                continue;
            }

            positions.push((nx, ny));
            let mut new_room = Room::new(nx, ny);
            new_room.add_obstacles();
            new_room.spawn_enemies(floor);

            // Add doors
            let dir_to_parent = match (dx, dy) {
                (0, -1) => Direction::Down,
                (0, 1) => Direction::Up,
                (-1, 0) => Direction::Right,
                (1, 0) => Direction::Left,
                _ => Direction::Up,
            };
            new_room.add_door(dir_to_parent);
            rooms[parent_idx].add_door(match dir_to_parent {
                Direction::Up => Direction::Down,
                Direction::Down => Direction::Up,
                Direction::Left => Direction::Right,
                Direction::Right => Direction::Left,
            });

            rooms.push(new_room);
        }

        // Boss room (last room)
        if let Some(last_room) = rooms.last_mut() {
            last_room.enemies.clear();
            let cx = SCREEN_WIDTH / 2.0;
            let cy = (SCREEN_HEIGHT - 64.0) / 2.0 + 32.0;
            last_room.enemies.push(Enemy::new(cx, cy, EnemyType::Boss));
        }

        Self {
            rooms,
            current_room: 0,
            floor,
        }
    }

    fn current(&self) -> &Room {
        &self.rooms[self.current_room]
    }

    fn current_mut(&mut self) -> &mut Room {
        &mut self.rooms[self.current_room]
    }

    fn find_room(&self, x: i32, y: i32) -> Option<usize> {
        self.rooms.iter().position(|r| r.x == x && r.y == y)
    }
}

// ===== RL INTERFACE =====
#[derive(Clone, serde::Serialize)]
struct Observation {
    player_x: f32,
    player_y: f32,
    player_hp: i32,
    enemies: Vec<(f32, f32, f32)>,  // x, y, hp
    tears: Vec<(f32, f32, bool)>,   // x, y, from_player
    nearest_enemy_dx: f32,
    nearest_enemy_dy: f32,
    room_cleared: bool,
}

#[derive(Clone, Copy, PartialEq)]
enum RLAction {
    Noop,
    MoveUp,
    MoveDown,
    MoveLeft,
    MoveRight,
    MoveUpLeft,
    MoveUpRight,
    MoveDownLeft,
    MoveDownRight,
    ShootUp,
    ShootDown,
    ShootLeft,
    ShootRight,
}

impl RLAction {
    fn from_id(id: usize) -> Self {
        match id {
            0 => RLAction::Noop,
            1 => RLAction::MoveUp,
            2 => RLAction::MoveDown,
            3 => RLAction::MoveLeft,
            4 => RLAction::MoveRight,
            5 => RLAction::MoveUpLeft,
            6 => RLAction::MoveUpRight,
            7 => RLAction::MoveDownLeft,
            8 => RLAction::MoveDownRight,
            9 => RLAction::ShootUp,
            10 => RLAction::ShootDown,
            11 => RLAction::ShootLeft,
            12 => RLAction::ShootRight,
            _ => RLAction::Noop,
        }
    }
}

// ===== GAME =====
struct Game {
    player: Player,
    dungeon: Dungeon,
    tears: Vec<Tear>,
    mode: GameMode,
    current_action: RLAction,
    score: i32,
    game_over: bool,
    victory: bool,
    frame: u64,
}

impl Game {
    fn new() -> Self {
        Self {
            player: Player::new(),
            dungeon: Dungeon::generate(1),
            tears: Vec::new(),
            mode: GameMode::Human,
            current_action: RLAction::Noop,
            score: 0,
            game_over: false,
            victory: false,
            frame: 0,
        }
    }

    fn reset(&mut self) {
        self.player = Player::new();
        self.dungeon = Dungeon::generate(1);
        self.tears.clear();
        self.score = 0;
        self.game_over = false;
        self.victory = false;
        self.frame = 0;
    }

    fn get_observation(&self) -> Observation {
        let room = self.dungeon.current();
        let mut enemies: Vec<(f32, f32, f32)> = room
            .enemies
            .iter()
            .map(|e| (e.x, e.y, e.hp))
            .collect();

        // Sort by distance to player
        enemies.sort_by(|a, b| {
            let da = (a.0 - self.player.x).powi(2) + (a.1 - self.player.y).powi(2);
            let db = (b.0 - self.player.x).powi(2) + (b.1 - self.player.y).powi(2);
            da.partial_cmp(&db).unwrap()
        });

        let (nearest_dx, nearest_dy) = if let Some(e) = enemies.first() {
            (e.0 - self.player.x, e.1 - self.player.y)
        } else {
            (0.0, 0.0)
        };

        Observation {
            player_x: self.player.x / SCREEN_WIDTH,
            player_y: self.player.y / SCREEN_HEIGHT,
            player_hp: self.player.hp,
            enemies: enemies.into_iter().take(5).collect(),
            tears: self.tears.iter().map(|t| (t.x, t.y, t.from_player)).collect(),
            nearest_enemy_dx: nearest_dx / SCREEN_WIDTH,
            nearest_enemy_dy: nearest_dy / SCREEN_HEIGHT,
            room_cleared: room.cleared,
        }
    }

    fn step(&mut self, action: RLAction) -> (Observation, f32, bool) {
        self.current_action = action;
        let old_hp = self.player.hp;
        let old_enemies = self.dungeon.current().enemies.len();

        self.update(1.0 / 60.0);

        // Calculate reward
        let mut reward = 0.0;

        // Reward for killing enemies
        let new_enemies = self.dungeon.current().enemies.len();
        reward += (old_enemies as i32 - new_enemies as i32) as f32 * 10.0;

        // Penalty for taking damage
        reward += (self.player.hp - old_hp) as f32 * 5.0;

        // Small reward for survival
        reward += 0.01;

        // Big reward for clearing room
        if self.dungeon.current().cleared && !self.dungeon.current().enemies.is_empty() {
            reward += 50.0;
        }

        // Victory bonus
        if self.victory {
            reward += 100.0;
        }

        // Game over penalty
        if self.game_over {
            reward -= 50.0;
        }

        let done = self.game_over || self.victory;
        (self.get_observation(), reward, done)
    }

    fn handle_input(&mut self) {
        if self.mode == GameMode::Human {
            // Movement
            let mut dx = 0.0;
            let mut dy = 0.0;
            if is_key_down(KeyCode::W) || is_key_down(KeyCode::Up) { dy -= 1.0; }
            if is_key_down(KeyCode::S) || is_key_down(KeyCode::Down) { dy += 1.0; }
            if is_key_down(KeyCode::A) || is_key_down(KeyCode::Left) { dx -= 1.0; }
            if is_key_down(KeyCode::D) || is_key_down(KeyCode::Right) { dx += 1.0; }

            self.player.vx = dx * self.player.speed;
            self.player.vy = dy * self.player.speed;

            // Shooting (Arrow keys or IJKL)
            if is_key_down(KeyCode::I) { self.try_shoot(Direction::Up); }
            if is_key_down(KeyCode::K) { self.try_shoot(Direction::Down); }
            if is_key_down(KeyCode::J) { self.try_shoot(Direction::Left); }
            if is_key_down(KeyCode::L) { self.try_shoot(Direction::Right); }
        } else {
            // RL mode - apply action
            self.apply_action(self.current_action);
        }

        // Reset
        if is_key_pressed(KeyCode::R) {
            self.reset();
        }
    }

    fn apply_action(&mut self, action: RLAction) {
        self.player.vx = 0.0;
        self.player.vy = 0.0;

        match action {
            RLAction::Noop => {}
            RLAction::MoveUp => self.player.vy = -self.player.speed,
            RLAction::MoveDown => self.player.vy = self.player.speed,
            RLAction::MoveLeft => self.player.vx = -self.player.speed,
            RLAction::MoveRight => self.player.vx = self.player.speed,
            RLAction::MoveUpLeft => {
                self.player.vx = -self.player.speed * 0.707;
                self.player.vy = -self.player.speed * 0.707;
            }
            RLAction::MoveUpRight => {
                self.player.vx = self.player.speed * 0.707;
                self.player.vy = -self.player.speed * 0.707;
            }
            RLAction::MoveDownLeft => {
                self.player.vx = -self.player.speed * 0.707;
                self.player.vy = self.player.speed * 0.707;
            }
            RLAction::MoveDownRight => {
                self.player.vx = self.player.speed * 0.707;
                self.player.vy = self.player.speed * 0.707;
            }
            RLAction::ShootUp => self.try_shoot(Direction::Up),
            RLAction::ShootDown => self.try_shoot(Direction::Down),
            RLAction::ShootLeft => self.try_shoot(Direction::Left),
            RLAction::ShootRight => self.try_shoot(Direction::Right),
        }
    }

    fn try_shoot(&mut self, dir: Direction) {
        if self.player.fire_timer <= 0.0 {
            let (vx, vy) = match dir {
                Direction::Up => (0.0, -self.player.tears_speed),
                Direction::Down => (0.0, self.player.tears_speed),
                Direction::Left => (-self.player.tears_speed, 0.0),
                Direction::Right => (self.player.tears_speed, 0.0),
            };
            self.tears.push(Tear {
                x: self.player.x,
                y: self.player.y,
                vx,
                vy,
                damage: self.player.damage,
                from_player: true,
                lifetime: 1.0,
            });
            self.player.fire_timer = self.player.fire_rate;
        }
    }

    fn update(&mut self, dt: f32) {
        if self.game_over || self.victory {
            return;
        }

        self.frame += 1;
        self.handle_input();

        // Update player position
        let new_x = self.player.x + self.player.vx * dt;
        let new_y = self.player.y + self.player.vy * dt;

        // Collision with walls/obstacles
        if self.can_move_to(new_x, self.player.y) {
            self.player.x = new_x;
        }
        if self.can_move_to(self.player.x, new_y) {
            self.player.y = new_y;
        }

        // Check door transitions
        self.check_doors();

        // Update timers
        self.player.fire_timer -= dt;
        self.player.invincible_timer -= dt;

        // Update tears
        self.update_tears(dt);

        // Update enemies
        self.update_enemies(dt);

        // Check collisions
        self.check_collisions();

        // Check room cleared
        let room = self.dungeon.current_mut();
        if room.enemies.is_empty() && !room.cleared {
            room.cleared = true;
            self.score += 100;

            // Check if boss room cleared
            if self.dungeon.current_room == self.dungeon.rooms.len() - 1 {
                self.victory = true;
            }
        }

        // Check game over
        if self.player.hp <= 0 {
            self.game_over = true;
        }
    }

    fn can_move_to(&self, x: f32, y: f32) -> bool {
        let margin = 12.0;
        let room = self.dungeon.current();

        // Check bounds
        if x < margin || x > SCREEN_WIDTH - margin {
            return false;
        }
        if y < 32.0 + margin || y > SCREEN_HEIGHT - margin {
            return false;
        }

        // Check tiles
        let tx = (x / TILE_SIZE) as usize;
        let ty = ((y - 32.0) / TILE_SIZE) as usize;

        if tx < ROOM_WIDTH && ty < ROOM_HEIGHT {
            match room.tiles[ty][tx] {
                TileType::Wall | TileType::Rock | TileType::Pit => false,
                TileType::Door(_) => room.cleared || room.enemies.is_empty(),
                _ => true,
            }
        } else {
            false
        }
    }

    fn check_doors(&mut self) {
        let room = self.dungeon.current();
        if !room.cleared && !room.enemies.is_empty() {
            return;
        }

        let tx = (self.player.x / TILE_SIZE) as usize;
        let ty = ((self.player.y - 32.0) / TILE_SIZE) as usize;

        if tx < ROOM_WIDTH && ty < ROOM_HEIGHT {
            if let TileType::Door(dir) = room.tiles[ty][tx] {
                let (dx, dy) = match dir {
                    Direction::Up => (0, -1),
                    Direction::Down => (0, 1),
                    Direction::Left => (-1, 0),
                    Direction::Right => (1, 0),
                };

                let current = self.dungeon.current();
                let (new_rx, new_ry) = (current.x + dx, current.y + dy);

                if let Some(room_idx) = self.dungeon.find_room(new_rx, new_ry) {
                    self.dungeon.current_room = room_idx;
                    self.tears.clear();

                    // Move player to opposite side
                    match dir {
                        Direction::Up => self.player.y = SCREEN_HEIGHT - 64.0,
                        Direction::Down => self.player.y = 64.0,
                        Direction::Left => self.player.x = SCREEN_WIDTH - 48.0,
                        Direction::Right => self.player.x = 48.0,
                    }
                }
            }
        }
    }

    fn update_tears(&mut self, dt: f32) {
        for tear in &mut self.tears {
            tear.x += tear.vx * dt;
            tear.y += tear.vy * dt;
            tear.lifetime -= dt;
        }

        // Remove dead tears
        self.tears.retain(|t| {
            t.lifetime > 0.0
            && t.x > 0.0 && t.x < SCREEN_WIDTH
            && t.y > 32.0 && t.y < SCREEN_HEIGHT
        });
    }

    fn update_enemies(&mut self, dt: f32) {
        let player_x = self.player.x;
        let player_y = self.player.y;
        let mut new_tears = Vec::new();

        let enemies = &mut self.dungeon.current_mut().enemies;
        for enemy in enemies.iter_mut() {
            enemy.timer += dt;
            enemy.shoot_timer += dt;

            match enemy.enemy_type {
                EnemyType::Fly => {
                    // Random movement
                    if enemy.timer > 0.5 {
                        enemy.vx = mqrand::gen_range(-80.0, 80.0);
                        enemy.vy = mqrand::gen_range(-80.0, 80.0);
                        enemy.timer = 0.0;
                    }
                }
                EnemyType::Gaper => {
                    // Chase player
                    let dx = player_x - enemy.x;
                    let dy = player_y - enemy.y;
                    let dist = (dx * dx + dy * dy).sqrt();
                    if dist > 0.0 {
                        enemy.vx = dx / dist * 60.0;
                        enemy.vy = dy / dist * 60.0;
                    }
                }
                EnemyType::Shooter => {
                    // Shoot at player periodically
                    if enemy.shoot_timer > 2.0 {
                        let dx = player_x - enemy.x;
                        let dy = player_y - enemy.y;
                        let dist = (dx * dx + dy * dy).sqrt();
                        if dist > 0.0 {
                            new_tears.push(Tear {
                                x: enemy.x,
                                y: enemy.y,
                                vx: dx / dist * 150.0,
                                vy: dy / dist * 150.0,
                                damage: 1.0,
                                from_player: false,
                                lifetime: 2.0,
                            });
                        }
                        enemy.shoot_timer = 0.0;
                    }
                }
                EnemyType::Boss => {
                    // Shoot pattern
                    if enemy.shoot_timer > 1.0 {
                        for i in 0..8 {
                            let angle = (i as f32 / 8.0) * std::f32::consts::PI * 2.0
                                + enemy.timer * 0.5;
                            new_tears.push(Tear {
                                x: enemy.x,
                                y: enemy.y,
                                vx: angle.cos() * 120.0,
                                vy: angle.sin() * 120.0,
                                damage: 1.0,
                                from_player: false,
                                lifetime: 3.0,
                            });
                        }
                        enemy.shoot_timer = 0.0;
                    }

                    // Slow chase
                    let dx = player_x - enemy.x;
                    let dy = player_y - enemy.y;
                    let dist = (dx * dx + dy * dy).sqrt();
                    if dist > 0.0 {
                        enemy.vx = dx / dist * 30.0;
                        enemy.vy = dy / dist * 30.0;
                    }
                }
            }

            // Move enemy
            enemy.x += enemy.vx * dt;
            enemy.y += enemy.vy * dt;

            // Clamp to room
            enemy.x = enemy.x.clamp(48.0, SCREEN_WIDTH - 48.0);
            enemy.y = enemy.y.clamp(80.0, SCREEN_HEIGHT - 48.0);
        }

        self.tears.extend(new_tears);
    }

    fn check_collisions(&mut self) {
        let player_rect = self.player.rect();

        // Player tears vs enemies
        let enemies = &mut self.dungeon.current_mut().enemies;
        for tear in &mut self.tears {
            if !tear.from_player {
                continue;
            }
            for enemy in enemies.iter_mut() {
                if tear.rect().overlaps(&enemy.rect()) {
                    enemy.hp -= tear.damage;
                    tear.lifetime = 0.0;
                    break;
                }
            }
        }

        // Remove dead enemies
        let score_add = enemies.iter().filter(|e| e.hp <= 0.0).count() as i32 * 10;
        self.score += score_add;
        enemies.retain(|e| e.hp > 0.0);

        // Enemy tears vs player
        if self.player.invincible_timer <= 0.0 {
            for tear in &mut self.tears {
                if tear.from_player {
                    continue;
                }
                if tear.rect().overlaps(&player_rect) {
                    self.player.hp -= tear.damage as i32;
                    self.player.invincible_timer = 1.0;
                    tear.lifetime = 0.0;
                    break;
                }
            }
        }

        // Contact damage from enemies
        if self.player.invincible_timer <= 0.0 {
            let enemies = &self.dungeon.current().enemies;
            for enemy in enemies {
                if enemy.damage > 0.0 && enemy.rect().overlaps(&player_rect) {
                    self.player.hp -= enemy.damage as i32;
                    self.player.invincible_timer = 1.0;
                    break;
                }
            }
        }
    }

    fn draw(&self) {
        clear_background(Color::from_rgba(40, 40, 50, 255));

        // Draw room
        self.draw_room();

        // Draw tears
        for tear in &self.tears {
            let color = if tear.from_player {
                Color::from_rgba(100, 180, 255, 255)
            } else {
                Color::from_rgba(255, 100, 100, 255)
            };
            draw_circle(tear.x, tear.y, 6.0, color);
        }

        // Draw enemies
        for enemy in &self.dungeon.current().enemies {
            let (color, size) = match enemy.enemy_type {
                EnemyType::Fly => (Color::from_rgba(100, 100, 100, 255), 10.0),
                EnemyType::Gaper => (Color::from_rgba(200, 100, 100, 255), 14.0),
                EnemyType::Shooter => (Color::from_rgba(100, 200, 100, 255), 12.0),
                EnemyType::Boss => (Color::from_rgba(150, 50, 150, 255), 24.0),
            };
            draw_circle(enemy.x, enemy.y, size, color);

            // Health bar for boss
            if enemy.enemy_type == EnemyType::Boss {
                let bar_width = 60.0;
                let hp_pct = enemy.hp / enemy.max_hp;
                draw_rectangle(enemy.x - bar_width / 2.0, enemy.y - 35.0, bar_width, 6.0, DARKGRAY);
                draw_rectangle(enemy.x - bar_width / 2.0, enemy.y - 35.0, bar_width * hp_pct, 6.0, RED);
            }
        }

        // Draw player
        let player_color = if self.player.invincible_timer > 0.0 && (self.frame % 10) < 5 {
            Color::from_rgba(255, 255, 255, 128)
        } else {
            Color::from_rgba(255, 220, 180, 255)
        };
        draw_circle(self.player.x, self.player.y, 12.0, player_color);

        // Eyes
        draw_circle(self.player.x - 4.0, self.player.y - 2.0, 3.0, WHITE);
        draw_circle(self.player.x + 4.0, self.player.y - 2.0, 3.0, WHITE);
        draw_circle(self.player.x - 4.0, self.player.y - 2.0, 1.5, BLACK);
        draw_circle(self.player.x + 4.0, self.player.y - 2.0, 1.5, BLACK);

        // Draw HUD
        self.draw_hud();

        // Draw overlays
        if self.game_over {
            draw_rectangle(0.0, 0.0, SCREEN_WIDTH, SCREEN_HEIGHT, Color::from_rgba(0, 0, 0, 180));
            let text = "GAME OVER";
            let text_size = 40;
            let text_width = text.len() as f32 * text_size as f32 * 0.5;
            draw_text(text, SCREEN_WIDTH / 2.0 - text_width / 2.0, SCREEN_HEIGHT / 2.0, text_size as f32, RED);
            draw_text("Press R to restart", SCREEN_WIDTH / 2.0 - 80.0, SCREEN_HEIGHT / 2.0 + 40.0, 20.0, WHITE);
        }

        if self.victory {
            draw_rectangle(0.0, 0.0, SCREEN_WIDTH, SCREEN_HEIGHT, Color::from_rgba(0, 0, 0, 180));
            let text = "VICTORY!";
            draw_text(text, SCREEN_WIDTH / 2.0 - 80.0, SCREEN_HEIGHT / 2.0, 40.0, GOLD);
            draw_text(&format!("Score: {}", self.score), SCREEN_WIDTH / 2.0 - 50.0, SCREEN_HEIGHT / 2.0 + 40.0, 24.0, WHITE);
            draw_text("Press R to restart", SCREEN_WIDTH / 2.0 - 80.0, SCREEN_HEIGHT / 2.0 + 70.0, 20.0, WHITE);
        }
    }

    fn draw_room(&self) {
        let room = self.dungeon.current();

        for y in 0..ROOM_HEIGHT {
            for x in 0..ROOM_WIDTH {
                let px = x as f32 * TILE_SIZE;
                let py = y as f32 * TILE_SIZE + 32.0;

                match room.tiles[y][x] {
                    TileType::Floor => {
                        draw_rectangle(px, py, TILE_SIZE, TILE_SIZE, Color::from_rgba(60, 50, 45, 255));
                        // Subtle grid lines
                        draw_rectangle_lines(px, py, TILE_SIZE, TILE_SIZE, 1.0, Color::from_rgba(70, 60, 55, 255));
                    }
                    TileType::Wall => {
                        draw_rectangle(px, py, TILE_SIZE, TILE_SIZE, Color::from_rgba(80, 70, 60, 255));
                        draw_rectangle(px + 2.0, py + 2.0, TILE_SIZE - 4.0, TILE_SIZE - 4.0, Color::from_rgba(100, 90, 80, 255));
                    }
                    TileType::Door(dir) => {
                        let color = if room.cleared || room.enemies.is_empty() {
                            Color::from_rgba(150, 100, 50, 255)
                        } else {
                            Color::from_rgba(100, 60, 40, 255)
                        };
                        draw_rectangle(px, py, TILE_SIZE, TILE_SIZE, color);

                        // Door indicator
                        let (ox, oy, w, h) = match dir {
                            Direction::Up => (8.0, 0.0, 16.0, 8.0),
                            Direction::Down => (8.0, 24.0, 16.0, 8.0),
                            Direction::Left => (0.0, 8.0, 8.0, 16.0),
                            Direction::Right => (24.0, 8.0, 8.0, 16.0),
                        };
                        draw_rectangle(px + ox, py + oy, w, h, Color::from_rgba(50, 30, 20, 255));
                    }
                    TileType::Rock => {
                        draw_rectangle(px, py, TILE_SIZE, TILE_SIZE, Color::from_rgba(60, 50, 45, 255));
                        draw_circle(px + TILE_SIZE / 2.0, py + TILE_SIZE / 2.0, 12.0, Color::from_rgba(120, 110, 100, 255));
                    }
                    TileType::Pit => {
                        draw_rectangle(px, py, TILE_SIZE, TILE_SIZE, Color::from_rgba(20, 20, 25, 255));
                    }
                }
            }
        }
    }

    fn draw_hud(&self) {
        // HUD background
        draw_rectangle(0.0, 0.0, SCREEN_WIDTH, 32.0, Color::from_rgba(30, 30, 35, 255));

        // Hearts
        for i in 0..(self.player.max_hp / 2) {
            let x = 10.0 + i as f32 * 24.0;
            let y = 16.0;
            let full_hearts = self.player.hp / 2;
            let half = self.player.hp % 2;

            if i < full_hearts {
                // Full heart
                draw_circle(x, y, 8.0, RED);
            } else if i == full_hearts && half == 1 {
                // Half heart
                draw_circle(x, y, 8.0, Color::from_rgba(100, 50, 50, 255));
                draw_circle(x - 4.0, y, 4.0, RED);
            } else {
                // Empty heart
                draw_circle(x, y, 8.0, Color::from_rgba(60, 30, 30, 255));
            }
        }

        // Room indicator
        let room = self.dungeon.current();
        draw_text(
            &format!("Room: ({}, {})", room.x, room.y),
            SCREEN_WIDTH - 120.0, 20.0, 16.0, WHITE
        );

        // Score
        draw_text(&format!("Score: {}", self.score), SCREEN_WIDTH / 2.0 - 40.0, 20.0, 16.0, GOLD);

        // Mode indicator
        let mode_text = match self.mode {
            GameMode::Human => "HUMAN",
            GameMode::RL => "RL",
        };
        draw_text(mode_text, 10.0, SCREEN_HEIGHT - 10.0, 14.0, Color::from_rgba(100, 100, 100, 255));

        // Controls help
        draw_text("WASD: Move | IJKL: Shoot | R: Reset", SCREEN_WIDTH / 2.0 - 130.0, SCREEN_HEIGHT - 10.0, 14.0, Color::from_rgba(80, 80, 80, 255));
    }
}

// ===== MAIN =====
fn window_conf() -> Conf {
    Conf {
        window_title: "Mini Isaac".to_string(),
        window_width: SCREEN_WIDTH as i32,
        window_height: SCREEN_HEIGHT as i32,
        window_resizable: false,
        ..Default::default()
    }
}

#[macroquad::main(window_conf)]
async fn main() {
    let mut game = Game::new();

    loop {
        game.update(get_frame_time());
        game.draw();
        next_frame().await;
    }
}
